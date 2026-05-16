################################################################################
# App Runner — pulls the API image from ECR, talks to RDS over a VPC connector.
################################################################################

# IAM role App Runner assumes to pull from ECR.
data "aws_iam_policy_document" "apprunner_ecr_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${var.project_name}-apprunner-ecr"
  assume_role_policy = data.aws_iam_policy_document.apprunner_ecr_assume.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# Instance role — what the running container can do (read its DB secret).
data "aws_iam_policy_document" "apprunner_instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.project_name}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.apprunner_instance_assume.json
}

data "aws_iam_policy_document" "apprunner_instance" {
  statement {
    sid       = "ReadDbSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.db_url.arn]
  }
}

resource "aws_iam_role_policy" "apprunner_instance" {
  name   = "${var.project_name}-apprunner-instance"
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.apprunner_instance.json
}

# VPC connector — lets the App Runner instances reach the private RDS.
resource "aws_apprunner_vpc_connector" "main" {
  vpc_connector_name = var.project_name
  subnets            = data.aws_subnets.default.ids
  security_groups    = [aws_security_group.apprunner_vpc.id]
}

resource "aws_apprunner_service" "api" {
  service_name = var.project_name

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = "8000"

        runtime_environment_variables = {
          API_HOST       = "0.0.0.0"
          API_PORT       = "8000"
          LOG_LEVEL      = "INFO"
          API_MODEL_PATH = "/app/models/baseline.joblib"
        }

        runtime_environment_secrets = {
          POSTGRES_USER     = "${aws_secretsmanager_secret.db_url.arn}:POSTGRES_USER::"
          POSTGRES_PASSWORD = "${aws_secretsmanager_secret.db_url.arn}:POSTGRES_PASSWORD::"
          POSTGRES_DB       = "${aws_secretsmanager_secret.db_url.arn}:POSTGRES_DB::"
          POSTGRES_HOST     = "${aws_secretsmanager_secret.db_url.arn}:POSTGRES_HOST::"
          POSTGRES_PORT     = "${aws_secretsmanager_secret.db_url.arn}:POSTGRES_PORT::"
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.app_runner_cpu
    memory            = var.app_runner_memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  network_configuration {
    egress_configuration {
      egress_type       = "VPC"
      vpc_connector_arn = aws_apprunner_vpc_connector.main.arn
    }
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/healthz"
    interval            = 20
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  depends_on = [
    aws_iam_role_policy_attachment.apprunner_ecr,
    aws_db_instance.main,
    aws_secretsmanager_secret_version.db_url,
  ]
}
