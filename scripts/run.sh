#!/bin/bash

# Mario RL Docker Run Script
# This script provides easy commands to run the Mario RL containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SERVICE="mario-rl"
COMMAND=""
INTERACTIVE=true
DETACH=false
BUILD=false

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
Usage: $0 [OPTIONS] [COMMAND]

Run Mario RL containers with various options

OPTIONS:
    -s, --service SERVICE   Docker service to run (mario-rl|mario-rl-dev) [default: mario-rl]
    -d, --detach           Run in detached mode (background)
    -b, --build            Build images before running
    --no-interactive       Run without interactive mode
    -h, --help             Show this help message

COMMANDS:
    train                  Run training script
    train-100k             Run 100K episode training
    evaluate               Run evaluation
    visualize              Run visualization
    bash                   Open bash shell (default)
    jupyter                Start Jupyter Lab (dev service only)
    tensorboard           Start TensorBoard
    test                   Run tests

EXAMPLES:
    $0                                    # Start runtime container with bash
    $0 --service mario-rl-dev            # Start development container
    $0 train                             # Run basic training
    $0 train-100k                        # Run 100K episode training
    $0 --service mario-rl-dev jupyter    # Start Jupyter in dev container
    $0 -d --service mario-rl-dev         # Start dev container in background
    $0 --build train                     # Build and run training

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            INTERACTIVE=false
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        --no-interactive)
            INTERACTIVE=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        train|train-100k|evaluate|visualize|bash|jupyter|tensorboard|test)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate service
if [[ "$SERVICE" != "mario-rl" && "$SERVICE" != "mario-rl-dev" ]]; then
    print_error "Invalid service: $SERVICE"
    print_error "Valid services: mario-rl, mario-rl-dev"
    exit 1
fi

# Set default command
if [[ -z "$COMMAND" ]]; then
    COMMAND="bash"
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    print_error "docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

# Build if requested
if [[ "$BUILD" == "true" ]]; then
    print_status "Building images..."
    docker-compose build "$SERVICE"
fi

# Prepare docker-compose command
COMPOSE_ARGS=()

if [[ "$DETACH" == "true" ]]; then
    COMPOSE_ARGS+=("-d")
fi

# Define container commands
case $COMMAND in
    train)
        CONTAINER_CMD="python training_scripts/train_mario.py"
        ;;
    train-100k)
        CONTAINER_CMD="python training_scripts/train_mario_100k_simple.py"
        ;;
    evaluate)
        CONTAINER_CMD="python -m mario_rl.cli.evaluate"
        ;;
    visualize)
        CONTAINER_CMD="python -m mario_rl.cli.visualize"
        ;;
    test)
        CONTAINER_CMD="python -m pytest tests/ -v"
        ;;
    tensorboard)
        CONTAINER_CMD="tensorboard --logdir=logs --host=0.0.0.0 --port=6006"
        ;;
    jupyter)
        if [[ "$SERVICE" != "mario-rl-dev" ]]; then
            print_error "Jupyter is only available in the development service (mario-rl-dev)"
            exit 1
        fi
        # Use the default command from docker-compose for dev service
        CONTAINER_CMD=""
        ;;
    bash)
        CONTAINER_CMD="/bin/bash"
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        exit 1
        ;;
esac

print_status "Starting service: $SERVICE"
print_status "Command: $COMMAND"

if [[ "$DETACH" == "true" ]]; then
    print_status "Running in detached mode..."
    
    if [[ -n "$CONTAINER_CMD" ]]; then
        docker-compose run "${COMPOSE_ARGS[@]}" "$SERVICE" $CONTAINER_CMD
    else
        docker-compose up "${COMPOSE_ARGS[@]}" "$SERVICE"
    fi
    
    print_success "Container started in background"
    print_status "Check logs with: docker-compose logs -f $SERVICE"
    print_status "Stop with: docker-compose stop $SERVICE"
    
else
    print_status "Running interactively..."
    
    # Set up X11 forwarding for macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v xhost >/dev/null 2>&1; then
            xhost +localhost >/dev/null 2>&1 || true
        fi
        export DISPLAY=host.docker.internal:0
    fi
    
    if [[ -n "$CONTAINER_CMD" ]]; then
        if [[ "$INTERACTIVE" == "true" ]]; then
            docker-compose run --rm "$SERVICE" $CONTAINER_CMD
        else
            docker-compose run --rm --no-TTY "$SERVICE" $CONTAINER_CMD
        fi
    else
        docker-compose up "$SERVICE"
    fi
fi

print_success "Command completed successfully!"