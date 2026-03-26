#!/bin/bash
# Script to package and deploy the Lambda function

set -e

echo "Creating Lambda deployment package..."

# Create clean deployment directory
rm -rf lambda_package
mkdir lambda_package
cd lambda_package

echo "Creating python/ directory for Lambda Layer structure..."
mkdir -p python

# echo "Installing nost-tools for Lambda..."
# pip install ../nost-tools \
#     --platform manylinux2014_x86_64 \
#     --target python \
#     --implementation cp \
#     --python-version 3.12 \
#     --only-binary=:all: \
#     --upgrade

echo "Installing other dependencies for Lambda..."
pip install -r ../requirements.txt \
    --platform manylinux2014_x86_64 \
    --target python \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade

echo "Creating Lambda Layer package..."
zip -r lambda_layer.zip python -x "*.pyc" -x "*/__pycache__/*" -x ".git/*"

echo ""
echo "Creating Lambda Function package (just code + config)..."
cp ../lambda_function_nost.py lambda_function.py

# Copy the YAML configuration file if it exists
if [ -f ../sos/sos.yaml ]; then
    cp ../sos/sos.yaml .
    echo "sos.yaml copied"
else
    echo "Warning: sos.yaml not found, you'll need to configure it separately"
fi

zip lambda_function.zip lambda_function.py sos.yaml

echo ""
echo "=========================================="
echo "✓ Deployment packages created!"
echo "=========================================="
echo ""
echo "Lambda Layer: lambda_package/lambda_layer.zip"
echo "  - Contains: nost_tools + all dependencies in python/ directory"
echo "  - Upload as Lambda Layer"
echo ""
echo "Lambda Function: lambda_package/lambda_function.zip"
echo "  - Contains: lambda_function.py + sos.yaml"
echo "  - Upload as Lambda Function code"
echo ""
echo "Deployment Instructions:"
echo "  1. Create Lambda Layer:"
echo "     aws lambda publish-layer-version \\"
echo "       --layer-name nost-tools-dependencies \\"
echo "       --zip-file fileb://lambda_package/lambda_layer.zip \\"
echo "       --compatible-runtimes python3.12"
echo ""
echo "  2. Update Lambda Function code:"
echo "     aws lambda update-function-code \\"
echo "       --function-name YOUR_LAMBDA_FUNCTION_NAME \\"
echo "       --zip-file fileb://lambda_package/lambda_function.zip"
echo ""
echo "  3. Attach Layer to Function in AWS Console:"
echo "     Lambda → Your Function → Layers → Add Layer → Custom Layers"
