# Remote state backend. ``terraform init -backend-config=backend.hcl`` or
# fill in the values inline. The bucket / table come from the bootstrap stack.

terraform {
  backend "s3" {
    key    = "factory-sensor-anomaly/aws/terraform.tfstate"
    region = "ap-northeast-1"
    # bucket and dynamodb_table must be passed via -backend-config to avoid
    # hardcoding account-specific names.
    encrypt = true
  }
}
