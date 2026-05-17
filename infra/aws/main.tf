################################################################################
# Application stack: ECR + RDS + App Runner (via apprunner.tf) + billing alarms.
#
# Networking strategy:
#   - Use the default VPC. Default subnets are *technically* "public" (have an
#     IGW route) but RDS sets publicly_accessible=false so it never gets a
#     public IP. App Runner's VPC connector lives in the same subnets and
#     reaches RDS by private IP. Saves ~50 lines of VPC Terraform for a
#     personal-scale project.
################################################################################

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ECR is managed by the bootstrap stack so it exists before any CI run can
# push an image. We just look it up here.
data "aws_ecr_repository" "api" {
  name = var.project_name
}

# ----- RDS --------------------------------------------------------------------

resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!#$%*-_=+"
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_security_group" "db" {
  name        = "${var.project_name}-db"
  description = "Allow inbound Postgres only from App Runner VPC connector"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "Postgres from App Runner VPC connector"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.apprunner_vpc.id]
  }

  egress {
    description = "All egress allowed RDS uses no outbound paths"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "apprunner_vpc" {
  name        = "${var.project_name}-apprunner"
  description = "Egress-only group attached to the App Runner VPC connector"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "main" {
  identifier                   = var.project_name
  engine                       = "postgres"
  engine_version               = "16.14"
  instance_class               = var.db_instance_class
  allocated_storage            = var.db_allocated_storage_gb
  storage_type                 = "gp3"
  storage_encrypted            = true
  db_name                      = var.db_name
  username                     = var.db_username
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.main.name
  vpc_security_group_ids       = [aws_security_group.db.id]
  publicly_accessible          = false
  multi_az                     = false
  backup_retention_period      = 1
  skip_final_snapshot          = true
  apply_immediately            = true
  deletion_protection          = false
  performance_insights_enabled = false
  auto_minor_version_upgrade   = true
}

# ----- Secrets ----------------------------------------------------------------

resource "aws_secretsmanager_secret" "db_url" {
  name                    = "${var.project_name}/db-url"
  description             = "Postgres connection URL consumed by the API container."
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "db_url" {
  secret_id = aws_secretsmanager_secret.db_url.id
  secret_string = jsonencode({
    POSTGRES_USER     = var.db_username
    POSTGRES_PASSWORD = random_password.db.result
    POSTGRES_DB       = var.db_name
    POSTGRES_HOST     = aws_db_instance.main.address
    POSTGRES_PORT     = tostring(aws_db_instance.main.port)
  })
}

# ----- Billing alarms ---------------------------------------------------------
# Only created if the alarm_email is set, so a CI-only deploy doesn't fail when
# the operator hasn't provided one yet.

resource "aws_sns_topic" "billing" {
  count = var.alarm_email == "" ? 0 : 1
  name  = "${var.project_name}-billing"
}

resource "aws_sns_topic_subscription" "billing_email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.billing[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  count               = var.alarm_email == "" ? 0 : length(var.billing_alarm_thresholds_usd)
  alarm_name          = "${var.project_name}-billing-${var.billing_alarm_thresholds_usd[count.index]}usd"
  alarm_description   = "Estimated AWS charges have crossed $${var.billing_alarm_thresholds_usd[count.index]}."
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600 # 6h
  statistic           = "Maximum"
  threshold           = var.billing_alarm_thresholds_usd[count.index]
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.billing[0].arn]
  dimensions = {
    Currency = "USD"
  }
}
