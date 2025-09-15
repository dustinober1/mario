#!/bin/bash#!/bin/bash



# Mario RL Setup Script# Mario RL Docker Development Setup Script

# Simple setup for local development and training# This script sets up the complete development environment



set -eset -e



# Colors for output# Colors for output

GREEN='\033[0;32m'RED='\033[0;31m'

BLUE='\033[0;34m'GREEN='\033[0;32m'

RED='\033[0;31m'YELLOW='\033[1;33m'

NC='\033[0m' # No ColorBLUE='\033[0;34m'

NC='\033[0m' # No Color

print_status() {

    echo -e "${BLUE}[INFO]${NC} $1"# Function to print colored output

}print_status() {

    echo -e "${BLUE}[INFO]${NC} $1"

print_success() {}

    echo -e "${GREEN}[SUCCESS]${NC} $1"

}print_success() {

    echo -e "${GREEN}[SUCCESS]${NC} $1"

print_error() {}

    echo -e "${RED}[ERROR]${NC} $1"

}print_warning() {

    echo -e "${YELLOW}[WARNING]${NC} $1"

# Main setup function}

main() {

    print_status "Setting up Mario RL development environment..."print_error() {

        echo -e "${RED}[ERROR]${NC} $1"

    # Check Python version}

    if command -v python3 &> /dev/null; then

        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)print_header() {

        print_status "Found Python $PYTHON_VERSION"    echo -e "\n${BLUE}============================================${NC}"

        if [[ "$PYTHON_VERSION" < "3.8" ]]; then    echo -e "${BLUE} $1${NC}"

            print_error "Python 3.8+ required. Found $PYTHON_VERSION"    echo -e "${BLUE}============================================${NC}\n"

            exit 1}

        fi

    else# Function to check if command exists

        print_error "Python 3 not found. Please install Python 3.8+"command_exists() {

        exit 1    command -v "$1" >/dev/null 2>&1

    fi}

    

    # Create virtual environment# Function to check Docker installation

    if [[ ! -d ".venv" ]]; thencheck_docker() {

        print_status "Creating virtual environment..."    print_header "Checking Docker Installation"

        python3 -m venv .venv    

        print_success "Virtual environment created"    if ! command_exists docker; then

    else        print_error "Docker is not installed. Please install Docker first."

        print_status "Virtual environment already exists"        print_status "Visit: https://docs.docker.com/get-docker/"

    fi        exit 1

        fi

    # Activate virtual environment    

    print_status "Activating virtual environment..."    if ! command_exists docker-compose; then

    source .venv/bin/activate        print_error "Docker Compose is not installed. Please install Docker Compose first."

            print_status "Visit: https://docs.docker.com/compose/install/"

    # Upgrade pip        exit 1

    print_status "Upgrading pip..."    fi

    pip install --upgrade pip    

        # Check if Docker is running

    # Install dependencies    if ! docker info >/dev/null 2>&1; then

    print_status "Installing dependencies..."        print_error "Docker is not running. Please start Docker first."

    pip install -r requirements.txt        exit 1

        fi

    # Install package in development mode    

    print_status "Installing Mario RL package..."    print_success "Docker and Docker Compose are installed and running"

    pip install -e .    

        # Show versions

    print_success "Setup complete!"    print_status "Docker version: $(docker --version)"

    echo    print_status "Docker Compose version: $(docker-compose --version)"

    print_status "To get started:"}

    echo "  1. Activate the environment: source .venv/bin/activate"

    echo "  2. Run training: python3 training_scripts/train_mario_production.py"# Function to setup directories

    echo "  3. Record videos: python3 training_scripts/record_mario_video.py"setup_directories() {

    echo    print_header "Setting Up Project Directories"

    print_status "Apple Silicon users automatically get MPS GPU acceleration!"    

}    # Create necessary directories

    directories=(

# Run main function        "models"

main "$@"        "videos/training"
        "videos/evaluation"
        "logs"
        "data"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
    
    print_success "All directories are set up"
}

# Function to make scripts executable
setup_scripts() {
    print_header "Setting Up Scripts"
    
    scripts=(
        "scripts/build.sh"
        "scripts/run.sh"
        "scripts/setup.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            chmod +x "$script"
            print_status "Made executable: $script"
        else
            print_warning "Script not found: $script"
        fi
    done
    
    print_success "Scripts are set up"
}

# Function to build Docker images
build_images() {
    print_header "Building Docker Images"
    
    print_status "Building runtime image..."
    if ./scripts/build.sh --target runtime; then
        print_success "Runtime image built successfully"
    else
        print_error "Failed to build runtime image"
        exit 1
    fi
    
    print_status "Building development image..."
    if ./scripts/build.sh --target development; then
        print_success "Development image built successfully"
    else
        print_error "Failed to build development image"
        exit 1
    fi
    
    print_success "All Docker images built successfully"
}

# Function to test setup
test_setup() {
    print_header "Testing Setup"
    
    print_status "Testing runtime container..."
    if docker-compose run --rm mario-rl python -c "import torch; import stable_baselines3; print('Dependencies OK')"; then
        print_success "Runtime container test passed"
    else
        print_error "Runtime container test failed"
        exit 1
    fi
    
    print_status "Testing development container..."
    if docker-compose run --rm mario-rl-dev python -c "import torch; import stable_baselines3; import jupyter; print('Dev dependencies OK')"; then
        print_success "Development container test passed"
    else
        print_error "Development container test failed"
        exit 1
    fi
    
    print_success "All tests passed"
}

# Function to show usage instructions
show_instructions() {
    print_header "Setup Complete! Here's how to use your environment:"
    
    cat << EOF
${GREEN}Quick Start Commands:${NC}

${BLUE}Training:${NC}
  ./scripts/run.sh train                    # Basic training
  ./scripts/run.sh train-100k              # 100K episode training

${BLUE}Development:${NC}
  ./scripts/run.sh --service mario-rl-dev  # Interactive development
  ./scripts/run.sh --service mario-rl-dev jupyter  # Jupyter Lab

${BLUE}Utilities:${NC}
  ./scripts/run.sh evaluate                # Evaluate trained models
  ./scripts/run.sh test                    # Run tests
  ./scripts/run.sh tensorboard            # Start TensorBoard

${BLUE}Container Management:${NC}
  docker-compose up mario-rl               # Start runtime container
  docker-compose up mario-rl-dev           # Start development container
  docker-compose down                      # Stop all containers
  docker-compose logs -f mario-rl          # View logs

${BLUE}Rebuilding:${NC}
  ./scripts/build.sh --force               # Force rebuild images
  docker-compose build                     # Rebuild using compose

${YELLOW}Access Points:${NC}
  Jupyter Lab: http://localhost:8888
  TensorBoard: http://localhost:6006

${YELLOW}Important Notes:${NC}
  - Models are saved to ./models/
  - Videos are saved to ./videos/
  - Logs are saved to ./logs/
  - All data persists between container restarts

EOF
}

# Main execution
main() {
    print_header "Mario RL Docker Development Setup"
    
    # Check if we're in the right directory
    if [[ ! -f "docker-compose.yml" ]]; then
        print_error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    # Run setup steps
    check_docker
    setup_directories
    setup_scripts
    
    # Ask user if they want to build images
    read -p "Do you want to build Docker images now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_images
        test_setup
    else
        print_warning "Skipping image build. Run './scripts/build.sh' manually when ready."
    fi
    
    show_instructions
    
    print_success "Setup completed successfully!"
}

# Run main function
main "$@"