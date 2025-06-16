#!/usr/bin/env python3
"""
Quick Deployment Script for Master Thesis Financial Analysis API
Helps deploy the Flask API to Heroku with one command.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are met for deployment."""
    print("🔍 Checking deployment requirements...")
    
    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Check if heroku CLI is installed
    if not run_command("heroku --version", "Checking Heroku CLI installation"):
        print("❌ Heroku CLI is not installed.")
        print("📥 Please install from: https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    # Check if logged into Heroku
    result = subprocess.run("heroku auth:whoami", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Not logged into Heroku.")
        print("🔐 Please run: heroku login")
        return False
    
    print("✅ All requirements met!")
    return True

def setup_git():
    """Initialize git repository if needed."""
    if not os.path.exists('.git'):
        print("📁 Initializing Git repository...")
        if not run_command("git init", "Initializing Git repository"):
            return False
        
        if not run_command("git add .", "Adding files to Git"):
            return False
        
        if not run_command('git commit -m "Initial commit - Master Thesis Financial Analysis API"', "Creating initial commit"):
            return False
    else:
        print("✅ Git repository already exists")
        # Add any new files
        run_command("git add .", "Adding new files to Git")
        run_command('git commit -m "Update for deployment"', "Committing changes")
    
    return True

def deploy_to_heroku():
    """Deploy the application to Heroku."""
    print("\n🚀 Starting Heroku deployment...")
    
    # Get app name from user
    app_name = input("Enter your Heroku app name (or press Enter for auto-generated): ").strip()
    
    if app_name:
        create_command = f"heroku create {app_name}"
    else:
        create_command = "heroku create"
    
    # Create Heroku app
    if not run_command(create_command, "Creating Heroku app"):
        print("ℹ️  App might already exist, continuing...")
    
    # Set environment variables
    env_commands = [
        "heroku config:set FLASK_ENV=production",
        "heroku config:set SECRET_KEY=thesis-secret-key-change-in-production",
        "heroku config:set LOG_LEVEL=INFO"
    ]
    
    for cmd in env_commands:
        run_command(cmd, f"Setting environment variable: {cmd.split('=')[0].split()[-1]}")
    
    # Deploy to Heroku
    if not run_command("git push heroku main", "Deploying to Heroku"):
        # Try with master branch
        if not run_command("git push heroku master", "Deploying to Heroku (master branch)"):
            return False
    
    # Get app URL
    result = subprocess.run("heroku apps:info --json", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        import json
        app_info = json.loads(result.stdout)
        app_url = app_info.get('web_url', 'https://your-app.herokuapp.com')
        print(f"\n🎉 Deployment successful!")
        print(f"🌐 Your API is now live at: {app_url}")
        print(f"📚 API Documentation: {app_url}")
        print(f"🔗 Example endpoint: {app_url}api/status")
        
        # Open in browser
        open_browser = input("\nOpen in browser? (y/n): ").lower().strip()
        if open_browser == 'y':
            run_command("heroku open", "Opening app in browser")
    
    return True

def main():
    """Main deployment workflow."""
    print("🏦 Master Thesis Financial Analysis API - Heroku Deployment")
    print("=" * 60)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup git
    if not setup_git():
        print("❌ Git setup failed")
        sys.exit(1)
    
    # Deploy to Heroku
    if not deploy_to_heroku():
        print("❌ Deployment failed")
        sys.exit(1)
    
    print("\n🎓 Your Master Thesis API is now live on the internet!")
    print("📊 You can now:")
    print("   - Share the URL with your thesis supervisors")
    print("   - Demonstrate live functionality in your defense")
    print("   - Include the URL in your thesis documentation")
    print("   - Add it to your academic CV/portfolio")
    
    print("\n📋 Next steps:")
    print("   - Test all endpoints to ensure they work")
    print("   - Monitor logs with: heroku logs --tail")
    print("   - Scale if needed with: heroku ps:scale web=1")

if __name__ == "__main__":
    main()
