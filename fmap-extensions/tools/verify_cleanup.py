#!/usr/bin/env python3
"""
Verification script for FMAP codebase cleanup
Checks that all cleanup tasks were completed successfully
"""

import os
import glob
import sys
from pathlib import Path

def check_heuristic_naming():
    """Check that incorrect heuristic names have been removed"""
    print("Checking heuristic naming consistency...")

    incorrect_names = ["DTG_Only", "Inc_DTG_Only", "FF_Heuristic"]
    python_files = glob.glob("**/*.py", recursive=True)

    issues_found = []
    for file_path in python_files:
        if ".git" in file_path or file_path == "verify_cleanup.py":
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for incorrect_name in incorrect_names:
                    if incorrect_name in content:
                        issues_found.append(f"{file_path}: contains '{incorrect_name}'")
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    if issues_found:
        print("Issues found with heuristic naming:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("All heuristic names are correct")
        return True

def check_correct_heuristic_mapping():
    """Verify that the correct heuristic mapping is present"""
    print("\n🔍 Checking correct heuristic mapping...")
    
    expected_mapping = {
        1: "DTG",
        2: "DTG+Landmarks", 
        3: "Inc_DTG+Landmarks",
        4: "Centroids",
        5: "MCS"
    }
    
    files_to_check = [
        "fmap-extensions/experiments/heuristic_comparison_analysis.py",
        "fmap-extensions/experiments/generate_metrics_comparison_table.py"
    ]
    
    all_correct = True
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            all_correct = False
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check if all expected heuristic names are present
            for heur_id, heur_name in expected_mapping.items():
                if f'"{heur_name}"' not in content:
                    print(f"❌ {file_path}: Missing correct heuristic name '{heur_name}'")
                    all_correct = False
                    
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            all_correct = False
    
    if all_correct:
        print("✅ Correct heuristic mapping found in all files")
    
    return all_correct

def check_redundant_files():
    """Check that redundant files have been removed"""
    print("\n🔍 Checking for redundant files...")
    
    should_not_exist = [
        "./simple_fmap_stats.py",
        "./FMAP_original.jar"
    ]
    
    should_exist = [
        "./fmap-extensions/automation/simple_fmap_stats.py",
        "./fmap-original/FMAP_original.jar"
    ]
    
    all_good = True
    
    # Check files that should NOT exist
    for file_path in should_not_exist:
        if os.path.exists(file_path):
            print(f"❌ Redundant file still exists: {file_path}")
            all_good = False
    
    # Check files that SHOULD exist
    for file_path in should_exist:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            all_good = False
    
    if all_good:
        print("✅ No redundant files found, all required files present")
    
    return all_good

def check_requirements_structure():
    """Check requirements.txt structure"""
    print("\n🔍 Checking requirements.txt structure...")
    
    main_req = "./requirements.txt"
    exp_req = "./fmap-extensions/experiments/requirements.txt"
    
    all_good = True
    
    # Check main requirements exists
    if not os.path.exists(main_req):
        print(f"❌ Main requirements file missing: {main_req}")
        all_good = False
    else:
        with open(main_req, 'r') as f:
            content = f.read()
            if "# Core analysis dependencies" not in content:
                print("❌ Main requirements.txt missing organization comments")
                all_good = False
    
    # Check experiments requirements references main
    if not os.path.exists(exp_req):
        print(f"❌ Experiments requirements file missing: {exp_req}")
        all_good = False
    else:
        with open(exp_req, 'r') as f:
            content = f.read()
            if "../../requirements.txt" not in content:
                print("❌ Experiments requirements.txt doesn't reference main file")
                all_good = False
    
    if all_good:
        print("✅ Requirements.txt structure is correct")
    
    return all_good

def check_jar_files():
    """Check JAR file organization"""
    print("\n🔍 Checking JAR file organization...")
    
    jar_files = glob.glob("**/*.jar", recursive=True)
    jar_files = [f for f in jar_files if ".git" not in f]
    
    expected_jars = [
        "fmap-original/FMAP_original.jar",
        "fmap-extensions/FMAP.jar", 
        "fmap-extensions/FMAP_final.jar"
    ]
    
    # Normalize paths for comparison
    found_jars = [os.path.normpath(jar) for jar in jar_files]
    expected_jars_norm = [os.path.normpath(jar) for jar in expected_jars]
    
    all_good = True
    
    # Check for unexpected JARs
    for jar in found_jars:
        if jar not in expected_jars_norm:
            print(f"❌ Unexpected JAR file: {jar}")
            all_good = False
    
    # Check for missing expected JARs
    for jar in expected_jars_norm:
        if jar not in found_jars:
            print(f"❌ Missing expected JAR file: {jar}")
            all_good = False
    
    if all_good:
        print(f"✅ JAR file organization correct ({len(found_jars)} files)")
    
    return all_good

def main():
    """Run all verification checks"""
    print("FMAP Codebase Cleanup Verification")
    print("=" * 50)

    checks = [
        check_heuristic_naming,
        check_correct_heuristic_mapping,
        check_redundant_files,
        check_requirements_structure,
        check_jar_files
    ]

    results = []
    for check in checks:
        results.append(check())

    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"ALL CHECKS PASSED ({passed}/{total})")
        print("Codebase cleanup was successful!")
        return 0
    else:
        print(f"SOME CHECKS FAILED ({passed}/{total})")
        print("Please review and fix the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
