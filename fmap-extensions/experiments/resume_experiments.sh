#!/bin/bash

# FMAP Experiment Resume Script
# This script helps resume experiments from where they left off

echo "FMAP Experiment Resume Tool"
echo "=========================="

cd "$(dirname "$0")" || exit 1

# Check if we have the resume script
if [[ ! -f "experiment_runner_resume.py" ]]; then
    echo "Error: experiment_runner_resume.py not found"
    exit 1
fi

# Function to show help
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --status, -s        Show experiment status and gaps"
    echo "  --kill-java, -k     Kill existing Java processes"
    echo "  --resume [INDEX]    Resume from specific index (default: auto-detect)"
    echo "  --force-restart     Restart all experiments, ignoring completed ones"
    echo
    echo "Examples:"
    echo "  $0 --status                # Check current status"
    echo "  $0 --kill-java             # Kill Java processes"
    echo "  $0 --resume                # Auto-resume from first gap"
    echo "  $0 --resume 60             # Resume from experiment 60"
    echo "  $0 --force-restart         # Start over from scratch"
}

# Parse arguments
RESUME_INDEX=""
SHOW_STATUS=false
KILL_JAVA=false
FORCE_RESTART=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --status|-s)
            SHOW_STATUS=true
            shift
            ;;
        --kill-java|-k)
            KILL_JAVA=true
            shift
            ;;
        --resume)
            if [[ -n "$2" && "$2" != --* ]]; then
                RESUME_INDEX="$2"
                shift 2
            else
                RESUME_INDEX="auto"
                shift
            fi
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Kill Java processes if requested
if [[ "$KILL_JAVA" == "true" ]]; then
    echo "Killing existing Java processes..."
    python3 experiment_runner_resume.py --kill-java
    exit 0
fi

# Show status if requested
if [[ "$SHOW_STATUS" == "true" ]]; then
    echo "Checking experiment status..."
    python3 experiment_runner_resume.py --list-completed
    exit 0
fi

# If no specific action, default to showing status
if [[ -z "$RESUME_INDEX" && "$FORCE_RESTART" == "false" ]]; then
    echo "No action specified. Showing status..."
    echo "Use --resume to continue experiments or --help for options."
    echo
    python3 experiment_runner_resume.py --list-completed
    exit 0
fi

# Resume experiments
if [[ "$FORCE_RESTART" == "true" ]]; then
    echo "Force restarting all experiments..."
    echo "This will ignore all previously completed experiments."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    python3 experiment_runner_resume.py --force-restart
    
elif [[ "$RESUME_INDEX" == "auto" ]]; then
    echo "Auto-detecting resume point..."
    
    # Get the first missing experiment index
    FIRST_MISSING=$(python3 experiment_runner_resume.py --list-completed | grep "Suggested resume from:" | awk '{print $4}')
    
    if [[ -n "$FIRST_MISSING" ]]; then
        echo "Resuming from experiment $FIRST_MISSING"
        python3 experiment_runner_resume.py --start-index "$FIRST_MISSING"
    else
        echo "No missing experiments found. All experiments may be complete."
        python3 experiment_runner_resume.py --list-completed
    fi
    
elif [[ -n "$RESUME_INDEX" ]]; then
    echo "Resuming from experiment $RESUME_INDEX"
    python3 experiment_runner_resume.py --start-index "$RESUME_INDEX"
fi

echo
echo "Experiment run completed!"
echo "Use --status to check final results." 