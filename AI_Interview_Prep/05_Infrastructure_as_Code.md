---
title: "Infrastructure as Code"
layout: default
parent: "DevOps & Cloud Infrastructure"
nav_order: 5
---

## SECTION 5: INFRASTRUCTURE AS CODE FOR AI - QUESTIONS & ANSWERS (2025-2026)


## 5.1  TERRAFORM FOR AI INFRASTRUCTURE


Q1: How do you use Terraform to provision AI/ML infrastructure on AWS?

Answer:
Terraform enables declarative, version-controlled infrastructure for ML.

Example - Complete SageMaker inference infrastructure:

  # provider.tf
  terraform {
    required_version = ">= 1.5"
    required_providers {
      aws = {
        source  = "hashicorp/aws"
        version = "~> 5.0"
      }
    }
    backend "s3" {
      bucket         = "ml-terraform-state"
      key            = "inference/terraform.tfstate"
      region         = "us-east-1"
      dynamodb_table = "terraform-locks"
      encrypt        = true
    }
  }

  # variables.tf
  variable "model_name" {
    description = "Name of the ML model"
    type        = string
  }
  variable "instance_type" {
    description = "SageMaker instance type"
    type        = string
    default     = "ml.g5.xlarge"
  }
  variable "instance_count" {
    description = "Number of instances"
    type        = number
    default     = 1
  }
  variable "model_data_url" {
    description = "S3 URL of model artifact"
    type        = string
  }

  # iam.tf
  resource "aws_iam_role" "sagemaker_execution" {
    name = "${var.model_name}-sagemaker-role"
    assume_role_policy = jsonencode({
      Version = "2012-10-17"
      Statement = [{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = { Service = "sagemaker.amazonaws.com" }
      }]
    })
  }

  resource "aws_iam_role_policy_attachment" "sagemaker_full" {
    role       = aws_iam_role.sagemaker_execution.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  }

  resource "aws_iam_role_policy" "s3_access" {
    name = "s3-model-access"
    role = aws_iam_role.sagemaker_execution.id
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [{
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::ml-models-bucket",
          "arn:aws:s3:::ml-models-bucket/*"
        ]
      }]
    })
  }

  # sagemaker.tf
  resource "aws_sagemaker_model" "model" {
    name               = var.model_name
    execution_role_arn = aws_iam_role.sagemaker_execution.arn

    primary_container {
      image          = "763104351884.dkr.ecr.us-east-1.amazonaws.com/\
                        pytorch-inference:2.2.0-gpu-py311-cu121-ubuntu22.04"
      model_data_url = var.model_data_url
      environment = {
        SAGEMAKER_PROGRAM = "inference.py"
      }
    }
  }

  resource "aws_sagemaker_endpoint_configuration" "config" {
    name = "${var.model_name}-config"

    production_variants {
      variant_name           = "primary"
      model_name             = aws_sagemaker_model.model.name
      instance_type          = var.instance_type
      initial_instance_count = var.instance_count
      initial_variant_weight = 1.0
    }

    data_capture_config {
      enable_capture              = true
      initial_sampling_percentage = 100
      destination_s3_uri          = "s3://ml-monitoring/${var.model_name}/capture"
      capture_options {
        capture_mode = "Input"
      }
      capture_options {
        capture_mode = "Output"
      }
    }
  }

  resource "aws_sagemaker_endpoint" "endpoint" {
    name                 = "${var.model_name}-endpoint"
    endpoint_config_name = aws_sagemaker_endpoint_configuration.config.name
  }

  # auto_scaling.tf
  resource "aws_appautoscaling_target" "sagemaker" {
    max_capacity       = 10
    min_capacity       = var.instance_count
    resource_id        = "endpoint/${aws_sagemaker_endpoint.endpoint.name}/\
                          variant/primary"
    scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
    service_namespace  = "sagemaker"
  }

  resource "aws_appautoscaling_policy" "sagemaker" {
    name               = "${var.model_name}-scaling"
    policy_type        = "TargetTrackingScaling"
    resource_id        = aws_appautoscaling_target.sagemaker.resource_id
    scalable_dimension = aws_appautoscaling_target.sagemaker.scalable_dimension
    service_namespace  = aws_appautoscaling_target.sagemaker.service_namespace

    target_tracking_scaling_policy_configuration {
      target_value = 70.0
      predefined_metric_specification {
        predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
      }
      scale_in_cooldown  = 300
      scale_out_cooldown = 60
    }
  }

  # monitoring.tf
  resource "aws_cloudwatch_metric_alarm" "endpoint_errors" {
    alarm_name          = "${var.model_name}-high-error-rate"
    comparison_operator = "GreaterThanThreshold"
    evaluation_periods  = 3
    metric_name         = "Invocation5XXErrors"
    namespace           = "AWS/SageMaker"
    period              = 60
    statistic           = "Sum"
    threshold           = 10
    alarm_actions       = [aws_sns_topic.ml_alerts.arn]
    dimensions = {
      EndpointName = aws_sagemaker_endpoint.endpoint.name
      VariantName  = "primary"
    }
  }

  resource "aws_sns_topic" "ml_alerts" {
    name = "${var.model_name}-alerts"
  }

  # outputs.tf
  output "endpoint_name" {
    value = aws_sagemaker_endpoint.endpoint.name
  }
  output "endpoint_arn" {
    value = aws_sagemaker_endpoint.endpoint.arn
  }

-------------------------------------------------------------------------------

Q2: How do you manage Terraform state for ML infrastructure?

Answer:
State management best practices for ML:

1. Remote state backend:
   - S3 + DynamoDB (AWS): State file in S3, locking via DynamoDB.
   - Azure Blob Storage: State file with blob lease locking.
   - Terraform Cloud: Managed state with UI, collaboration features.

2. State isolation:
   - Separate state per environment: dev/staging/prod.
   - Separate state per component: networking, compute, ML endpoints.
   - Use workspaces or directory structure:

   infrastructure/
     networking/          # VPC, subnets, security groups
       main.tf
       terraform.tfstate
     training/            # Training clusters, notebooks
       main.tf
       terraform.tfstate
     inference/           # Endpoints, auto-scaling
       main.tf
       terraform.tfstate
     monitoring/          # CloudWatch, alerts
       main.tf
       terraform.tfstate

3. State locking: Prevents concurrent modifications.
   DynamoDB table with LockID partition key.

4. Sensitive data:
   - Mark variables as sensitive: variable "api_key" { sensitive = true }
   - State file contains all values in plain text. Encrypt S3 bucket.
   - Use AWS Secrets Manager or HashiCorp Vault for secrets.

-------------------------------------------------------------------------------

Q3: How do you provision GPU infrastructure with Terraform?

Answer:
Example - EKS cluster with GPU node groups:

  module "eks" {
    source  = "terraform-aws-modules/eks/aws"
    version = "~> 20.0"

    cluster_name    = "ml-cluster"
    cluster_version = "1.29"
    vpc_id          = module.vpc.vpc_id
    subnet_ids      = module.vpc.private_subnets

    # CPU node group for general workloads
    eks_managed_node_groups = {
      cpu_workers = {
        instance_types = ["m5.2xlarge"]
        min_size       = 2
        max_size       = 10
        desired_size   = 3
      }

      # GPU node group for inference
      gpu_inference = {
        instance_types = ["g5.xlarge"]
        min_size       = 0
        max_size       = 5
        desired_size   = 1
        ami_type       = "AL2_x86_64_GPU"  # GPU-optimized AMI

        labels = {
          "nvidia.com/gpu.present" = "true"
          "workload-type"          = "inference"
        }
        taints = [{
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }]
      }

      # GPU node group for training (spot instances)
      gpu_training = {
        instance_types = ["p4d.24xlarge", "p3.16xlarge"]
        capacity_type  = "SPOT"
        min_size       = 0
        max_size       = 4
        desired_size   = 0
        ami_type       = "AL2_x86_64_GPU"

        labels = {
          "nvidia.com/gpu.present" = "true"
          "workload-type"          = "training"
        }
        taints = [{
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }]
      }
    }
  }

  # Install NVIDIA device plugin via Helm
  resource "helm_release" "nvidia_device_plugin" {
    name       = "nvidia-device-plugin"
    repository = "https://nvidia.github.io/k8s-device-plugin"
    chart      = "nvidia-device-plugin"
    namespace  = "kube-system"
    version    = "0.14.3"

    set {
      name  = "compatWithCPUManager"
      value = "true"
    }
  }

  # Install Karpenter for intelligent GPU auto-scaling
  resource "helm_release" "karpenter" {
    name       = "karpenter"
    repository = "oci://public.ecr.aws/karpenter"
    chart      = "karpenter"
    namespace  = "karpenter"
    version    = "0.33.0"
    # ... configuration ...
  }

## 5.2  CLOUDFORMATION FOR AI INFRASTRUCTURE


Q4: How do you use CloudFormation for AI/ML infrastructure?

Answer:
CloudFormation is AWS's native IaC service. Key differences from Terraform:
- AWS-only (no multi-cloud).
- Tighter integration with AWS services.
- Stack sets for multi-account/multi-region deployments.
- Drift detection built-in.
- No state file management (AWS manages state).

Example - SageMaker endpoint with CloudFormation:

  AWSTemplateFormatVersion: '2010-09-09'
  Description: SageMaker ML Inference Infrastructure

  Parameters:
    ModelName:
      Type: String
      Description: Name of the ML model
    ModelDataUrl:
      Type: String
      Description: S3 URI of model artifact
    InstanceType:
      Type: String
      Default: ml.g5.xlarge
      AllowedValues:
        - ml.g5.xlarge
        - ml.g5.2xlarge
        - ml.g5.4xlarge
        - ml.p4d.24xlarge
    InstanceCount:
      Type: Number
      Default: 1
      MinValue: 1
      MaxValue: 10

  Resources:
    SageMakerExecutionRole:
      Type: AWS::IAM::Role
      Properties:
        AssumeRolePolicyDocument:
          Version: '2012-10-17'
          Statement:
            - Effect: Allow
              Principal:
                Service: sagemaker.amazonaws.com
              Action: sts:AssumeRole
        ManagedPolicyArns:
          - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        Policies:
          - PolicyName: S3ModelAccess
            PolicyDocument:
              Version: '2012-10-17'
              Statement:
                - Effect: Allow
                  Action:
                    - s3:GetObject
                    - s3:ListBucket
                  Resource:
                    - arn:aws:s3:::ml-models-bucket
                    - arn:aws:s3:::ml-models-bucket/*

    SageMakerModel:
      Type: AWS::SageMaker::Model
      Properties:
        ModelName: !Ref ModelName
        ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
        PrimaryContainer:
          Image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/\
                 pytorch-inference:2.2.0-gpu-py311-cu121-ubuntu22.04
          ModelDataUrl: !Ref ModelDataUrl

    EndpointConfig:
      Type: AWS::SageMaker::EndpointConfig
      Properties:
        EndpointConfigName: !Sub '${ModelName}-config'
        ProductionVariants:
          - VariantName: primary
            ModelName: !GetAtt SageMakerModel.ModelName
            InstanceType: !Ref InstanceType
            InitialInstanceCount: !Ref InstanceCount
            InitialVariantWeight: 1.0

    Endpoint:
      Type: AWS::SageMaker::Endpoint
      Properties:
        EndpointName: !Sub '${ModelName}-endpoint'
        EndpointConfigName: !GetAtt EndpointConfig.EndpointConfigName

    ScalingTarget:
      Type: AWS::ApplicationAutoScaling::ScalableTarget
      Properties:
        MaxCapacity: 10
        MinCapacity: !Ref InstanceCount
        ResourceId: !Sub 'endpoint/${Endpoint.EndpointName}/variant/primary'
        ScalableDimension: 'sagemaker:variant:DesiredInstanceCount'
        ServiceNamespace: sagemaker
        RoleARN: !GetAtt SageMakerExecutionRole.Arn

    ScalingPolicy:
      Type: AWS::ApplicationAutoScaling::ScalingPolicy
      Properties:
        PolicyName: !Sub '${ModelName}-scaling'
        PolicyType: TargetTrackingScaling
        ScalingTargetId: !Ref ScalingTarget
        TargetTrackingScalingPolicyConfiguration:
          TargetValue: 70.0
          PredefinedMetricSpecification:
            PredefinedMetricType: SageMakerVariantInvocationsPerInstance
          ScaleInCooldown: 300
          ScaleOutCooldown: 60

  Outputs:
    EndpointName:
      Value: !GetAtt Endpoint.EndpointName
    EndpointArn:
      Value: !Ref Endpoint

Q5: Compare Terraform vs CloudFormation for ML infrastructure.

Answer:
  | Feature              | Terraform                     | CloudFormation              |
  |----------------------|-------------------------------|-----------------------------|
  | Multi-cloud          | Yes (AWS, Azure, GCP)         | AWS only                    |
  | State management     | You manage (S3, TF Cloud)     | AWS manages automatically   |
  | Language             | HCL (+ CDKTF for TypeScript)  | YAML/JSON (+ CDK for code) |
  | Drift detection      | terraform plan                | Built-in drift detection    |
  | Resource coverage    | Broad (community providers)   | AWS services only           |
  | Module ecosystem     | Terraform Registry            | Nested stacks, modules      |
  | Rollback             | Manual (apply previous state) | Automatic stack rollback    |
  | Import existing      | terraform import              | CloudFormation import       |
  | Testing              | terratest, terraform test     | cfn-lint, TaskCat           |
  | Secret management    | External (Vault, etc.)        | SSM Parameter Store, Secrets|
  | Speed                | Parallel by default           | Sequential (some parallel)  |
  | Learning curve       | Moderate                      | Moderate                    |

Recommendation for ML teams:
- Use Terraform if: Multi-cloud, need Kubernetes resources alongside AWS,
  team already knows Terraform, complex module composition.
- Use CloudFormation if: AWS-only, want automatic rollback, using CDK for
  programmatic infrastructure, tight integration with Service Catalog.
- Use CDK (AWS Cloud Development Kit) if: Want to define infrastructure in
  Python/TypeScript (natural for ML engineers who know Python).

CDK example for SageMaker (Python):
  from aws_cdk import (
      Stack, aws_sagemaker as sagemaker, aws_iam as iam
  )

  class MLInfraStack(Stack):
      def __init__(self, scope, construct_id, **kwargs):
          super().__init__(scope, construct_id, **kwargs)

          role = iam.Role(self, "SageMakerRole",
              assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
              managed_policies=[
                  iam.ManagedPolicy.from_aws_managed_policy_name(
                      "AmazonSageMakerFullAccess")
              ]
          )

          model = sagemaker.CfnModel(self, "Model",
              execution_role_arn=role.role_arn,
              primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                  image="...",
                  model_data_url="s3://..."
              )
          )
