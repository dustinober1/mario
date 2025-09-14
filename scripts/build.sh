#!/bin/bash

# Mario RL Docker Build Script
# This script builds the Docker images for the Mario RL project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TARGET="runtime"
PYTHON_VERSION="3.10"
FORCE_REBUILD=false
NO_CACHE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker images for Mario RL project

OPTIONS:
    -t, --target TARGET     Build target (runtime|development) [default: runtime]
    -p, --python VERSION    Python version (3.9|3.10|3.11) [default: 3.10]
    -f, --force            Force rebuild (remove existing images)
    --no-cache             Build without using cache
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Build runtime image with Python 3.10
    $0 --target development               # Build development image
    $0 --target runtime --python 3.11    # Build runtime with Python 3.11
    $0 --force --no-cache                 # Force rebuild without cache

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_REBUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate build target
if [[ "$BUILD_TARGET" != "runtime" && "$BUILD_TARGET" != "development" ]]; then
    print_error "Invalid build target: $BUILD_TARGET"
    print_error "Valid targets: runtime, development"
    exit 1
fi

# Validate Python version
if [[ "$PYTHON_VERSION" != "3.9" && "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    print_error "Invalid Python version: $PYTHON_VERSION"
    print_error "Valid versions: 3.9, 3.10, 3.11"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "Dockerfile" ]]; then
    print_error "Dockerfile not found. Please run this script from the project root."
    exit 1
fi

# Image names
IMAGE_NAME="mario-rl"
IMAGE_TAG="${BUILD_TARGET}-py${PYTHON_VERSION}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

print_status "Building Mario RL Docker image..."
print_status "Target: $BUILD_TARGET"
print_status "Python Version: $PYTHON_VERSION"
print_status "Image Name: $FULL_IMAGE_NAME"

# Force rebuild if requested
if [[ "$FORCE_REBUILD" == "true" ]]; then
    print_warning "Force rebuild requested - removing existing images..."
    
    # Remove specific image
    if docker image inspect "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
        print_status "Removing existing image: $FULL_IMAGE_NAME"
        docker rmi "$FULL_IMAGE_NAME" || true
    fi
    
    # Remove base images
    if docker image inspect "${IMAGE_NAME}:base-py${PYTHON_VERSION}" >/dev/null 2>&1; then
        print_status "Removing base image: ${IMAGE_NAME}:base-py${PYTHON_VERSION}"
        docker rmi "${IMAGE_NAME}:base-py${PYTHON_VERSION}" || true
    fi
fi

# Build arguments
BUILD_ARGS=(
    "--target" "$BUILD_TARGET"
    "--build-arg" "PYTHON_VERSION=$PYTHON_VERSION"
    "--tag" "$FULL_IMAGE_NAME"
)

# Add no-cache if requested
if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_ARGS+=("--no-cache")
    print_warning "Building without cache..."
fi

# Add progress output
BUILD_ARGS+=("--progress" "auto")

# Build the image
print_status "Starting Docker build..."
echo "Command: docker build ${BUILD_ARGS[*]} ."

if docker build "${BUILD_ARGS[@]}" .; then
    print_success "Successfully built image: $FULL_IMAGE_NAME"
    
    # Show image info
    print_status "Image details:"
    docker image inspect "$FULL_IMAGE_NAME" --format 'table {{.RepoTags}}\t{{.Size}}\t{{.Created}}' || true
    
    # Also tag as latest if this is runtime
    if [[ "$BUILD_TARGET" == "runtime" ]]; then
        docker tag "$FULL_IMAGE_NAME" "${IMAGE_NAME}:latest"
        print_success "Also tagged as: ${IMAGE_NAME}:latest"
    fi
    
    print_success "Build completed successfully!"
    print_status "You can now run the container with:"
    echo "  docker-compose up mario-rl"
    
    if [[ "$BUILD_TARGET" == "development" ]]; then
        echo "  docker-compose up mario-rl-dev"
    fi
    
else
    print_error "Build failed!"
    exit 1
fi