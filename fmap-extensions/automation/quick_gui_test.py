#!/usr/bin/env python3
"""
Quick FMAP GUI Test
A simple test script to validate GUI automation approach
"""

import os
import subprocess
import time
import pyautogui
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def test_gui_automation():
    """Test the GUI automation with a simple experiment"""
    
    print("FMAP GUI Automation Test")
    print("=" * 40)
    
    # Check prerequisites
    if not os.path.exists("FMAP.jar"):
        print("FMAP.jar not found")
        return False
    
    if not os.path.exists("Domains/driverlog/Pfile1"):
        print("Driverlog Pfile1 not found")
        return False
    
    print("Prerequisites check passed")
    
    # Configure pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 1.0
    
    # Get screen resolution
    screen_width, screen_height = pyautogui.size()
    print(f"🖥️  Screen resolution: {screen_width}x{screen_height}")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Test command for driverlog Pfile1 with DTG heuristic
        command = [
            "java", "-jar", "FMAP.jar",
            "driver1", "Domains/driverlog/Pfile1/DomainDriverlog.pddl", 
            "Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl",
            "driver2", "Domains/driverlog/Pfile1/DomainDriverlog.pddl", 
            "Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl",
            "Domains/driverlog/Pfile1/agents.txt",
            "-h", "1"  # DTG heuristic
        ]
        
        print("Starting FMAP with GUI...")
        print(f"Command: {' '.join(command)}")
        
        # Start FMAP process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Waiting for GUI to appear...")
        time.sleep(5)
        
        # Try to find any window (simplified approach)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Take initial screenshot
        print("📸 Capturing initial screenshot...")
        initial_screenshot = pyautogui.screenshot()
        initial_screenshot.save(f"results/test_initial_{timestamp}.png")
        
        # Wait a bit for planning to start
        print("⏳ Waiting for planning process...")
        time.sleep(10)
        
        # Take another screenshot
        print("📸 Capturing progress screenshot...")
        progress_screenshot = pyautogui.screenshot()
        progress_screenshot.save(f"results/test_progress_{timestamp}.png")
        
        # Wait for completion (or timeout)
        max_wait = 60  # 1 minute timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check if process is still running
            if process.poll() is not None:
                print("Process completed")
                break
            time.sleep(2)
        else:
            print("⏰ Timeout reached, terminating process")
            process.terminate()
        
        # Take final screenshot
        print("📸 Capturing final screenshot...")
        final_screenshot = pyautogui.screenshot()
        final_screenshot.save(f"results/test_final_{timestamp}.png")
        
        # Get process output
        try:
            stdout, stderr = process.communicate(timeout=5)
            output = stdout + stderr
            
            # Save output
            with open(f"results/test_output_{timestamp}.txt", "w") as f:
                f.write("=== FMAP GUI Test Output ===\n")
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
            
            print("Output saved to file")
            
            # Simple analysis
            success_indicators = [
                "Solution found", "Planning completed", "CoDMAP", 
                "Total time:", "Plan length:"
            ]
            
            success = any(indicator in output for indicator in success_indicators)
            print(f"Success detected: {success}")
            
            if success:
                print("GUI automation test PASSED")
                return True
            else:
                print("⚠️  No clear success indicators found")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  Process communication timeout")
            process.kill()
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        logging.error(f"GUI test failed: {e}")
        return False
    
    finally:
        # Cleanup any remaining processes
        try:
            if 'process' in locals():
                process.terminate()
            
            # Kill any java processes running FMAP
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'java' in proc.info['name'] and 'FMAP.jar' in ' '.join(proc.info['cmdline'] or []):
                        proc.kill()
                except:
                    pass
        except:
            pass
        
        print("🧹 Cleanup completed")


def main():
    """Main test function"""
    
    print("FMAP GUI Automation Quick Test")
    print("This will test basic GUI automation functionality")
    print("=" * 50)
    
    success = test_gui_automation()
    
    print("\n" + "=" * 50)
    if success:
        print("GUI automation test SUCCESSFUL!")
        print("You can now run the full automation suite")
        print("Use: python fmap_gui_automation.py")
    else:
        print("GUI automation test FAILED")
        print("Check the screenshots and output in results/ directory")
        print("You may need to adjust the automation approach")
    print("=" * 50)


if __name__ == "__main__":
    main() 