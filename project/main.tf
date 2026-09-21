terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket = "overcat-tf-state-651323680511"
    key = "fineweb-project/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "ssh_cidr_blocks" {
  description = "CIDR ranges allowed to reach port 22. Defaults to the whole internet; key-only auth is the control. Narrow to [\"x.x.x.x/32\"] if you prefer."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "public_key_path" {
  description = "Path to the local SSH public key uploaded to AWS."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "instance_type" {
  description = "EC2 instance type for the dataloader."
  type        = string
  default     = "c5a.xlarge"
}

variable "volume_size" {
  description = "Root volume size in GB."
  type        = number
  default     = 30
}

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  name = "fineweb-dataloader-${random_id.suffix.hex}"
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "fineweb_data" {
  bucket = "my-fineweb-2-5b-data-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_public_access_block" "fineweb_data" {
  bucket                  = aws_s3_bucket.fineweb_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "fineweb_data" {
  bucket = aws_s3_bucket.fineweb_data.id

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

resource "aws_iam_role" "dataloader" {
  name = local.name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "s3_access" {
  name = "s3-access-fineweb-bucket"
  role = aws_iam_role.dataloader.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:GetBucketLocation",
        ]
        Resource = aws_s3_bucket.fineweb_data.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts",
        ]
        Resource = "${aws_s3_bucket.fineweb_data.arn}/*"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "dataloader" {
  name = local.name
  role = aws_iam_role.dataloader.name
}

# ---------------------------------------------------------------------------
# Network lookups
# ---------------------------------------------------------------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_instance_type_offerings" "supported" {
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.supported.locations
  }
}

data "aws_ssm_parameter" "ubuntu" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

resource "aws_key_pair" "dataloader" {
  key_name   = local.name
  public_key = file(pathexpand(var.public_key_path))
}

resource "aws_security_group" "dataloader" {
  name        = local.name
  description = "SSH ingress, unrestricted egress"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH, key auth only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "dataloader" {
  ami                         = data.aws_ssm_parameter.ubuntu.value
  instance_type               = var.instance_type
  iam_instance_profile        = aws_iam_instance_profile.dataloader.name
  vpc_security_group_ids      = [aws_security_group.dataloader.id]
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  associate_public_ip_address = true
  key_name                    = aws_key_pair.dataloader.key_name

  root_block_device {
    volume_size           = var.volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = { Name = "fineweb-dataloader" }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "bucket_name" {
  description = "Target bucket for the downloaded shards."
  value       = aws_s3_bucket.fineweb_data.bucket
}

output "instance_id" {
  value = aws_instance.dataloader.id
}

output "public_ip" {
  value = aws_instance.dataloader.public_ip
}

output "availability_zone" {
  value = aws_instance.dataloader.availability_zone
}

output "ssh_command" {
  description = "Copy and run to connect."
  value       = "ssh ubuntu@${aws_instance.dataloader.public_ip}"
}