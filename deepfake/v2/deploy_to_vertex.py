import subprocess

# --- CONFIGURATION ---
REGION = "us-central1"
MODEL_NAME = "infer"           # The display name of your uploaded model
ENDPOINT_NAME = "infer-endpoint"
DEPLOYMENT_NAME = "infer-deployment"
MACHINE_TYPE = "n1-standard-4"
MIN_REPLICAS = 1
MAX_REPLICAS = 1

def run_cmd(cmd):
    """Run a shell command and stream output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    # 1. Create endpoint
    run_cmd(f'gcloud ai endpoints create --region={REGION} --display-name={ENDPOINT_NAME}')

    # 2. Get the latest endpoint ID
    endpoint_list = subprocess.check_output(
        f'gcloud ai endpoints list --region={REGION} --format="value(name)"',
        shell=True
    ).decode().strip().split("\n")
    endpoint_id = endpoint_list[-1]
    print(f"Using Endpoint ID: {endpoint_id}")

    # 3. Get the latest model ID
    model_list = subprocess.check_output(
        f'gcloud ai models list --region={REGION} --filter="displayName={MODEL_NAME}" --format="value(name)"',
        shell=True
    ).decode().strip().split("\n")
    model_id = model_list[-1]
    print(f"Using Model ID: {model_id}")

    # 4. Deploy model to endpoint
    run_cmd(
        f'gcloud ai endpoints deploy-model {endpoint_id} '
        f'--region={REGION} '
        f'--model={model_id} '
        f'--display-name={DEPLOYMENT_NAME} '
        f'--machine-type={MACHINE_TYPE} '
        f'--min-replica-count={MIN_REPLICAS} '
        f'--max-replica-count={MAX_REPLICAS}'
    )

if __name__ == "__main__":
    main()
