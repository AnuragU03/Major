"""
Quick Test Runner for Chapter 5 Results Generation

This script runs a minimal comparison test and generates Chapter 5 results.
Perfect for demonstration and validation.

Usage:
    python run_chapter5_test.py
"""

import os
import sys
import subprocess
from datetime import datetime

def run_quick_test():
    """Run a quick comparison test with minimal products"""
    print("\n" + "=" * 60)
    print("   QUICK CHAPTER 5 TEST RUNNER")
    print("=" * 60)
    
    print(f"\n⏱️  Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check if comparison_test.py exists
    if not os.path.exists('comparison_test.py'):
        print("❌ comparison_test.py not found!")
        return False
    
    # Check if analyze_pkl_results.py exists
    if not os.path.exists('analyze_pkl_results.py'):
        print("❌ analyze_pkl_results.py not found!")
        return False
    
    print("\n📋 Test Configuration:")
    print("   • Mode: Multi-Agent only (faster)")
    print("   • Products per query: 1")
    print("   • Total queries: 40 (10 categories × 4 queries)")
    print("   • Estimated time: 30-60 minutes")
    
    # Ask user confirmation
    choice = input("\n🚀 Start test? (y/n): ").strip().lower()
    if choice != 'y':
        print("Test cancelled.")
        return False
    
    try:
        # Run comparison test (multi-agent only)
        print("\n" + "=" * 60)
        print("   RUNNING COMPARISON TEST")
        print("=" * 60)
        
        # Import and run directly to avoid subprocess issues
        from comparison_test import run_multi_agent_test
        
        print("\n🤖 Running Multi-Agent test...")
        products, metrics = run_multi_agent_test()
        
        print(f"\n✅ Test completed!")
        print(f"   Products scraped: {len(products)}")
        print(f"   Time taken: {metrics.get('total_time_minutes', 0):.1f} minutes")
        
        # Generate Chapter 5 results
        print("\n" + "=" * 60)
        print("   GENERATING CHAPTER 5 RESULTS")
        print("=" * 60)
        
        from analyze_pkl_results import main as analyze_main
        analyze_main()
        
        print("\n🎉 SUCCESS! Chapter 5 results generated.")
        print("\n📁 Generated Files:")
        
        files_to_check = [
            'scraped_multi_agent.pkl',
            'CHAPTER_5_RESULTS.md',
            'performance_analysis.png',
            'umap_multi_agent.png'
        ]
        
        for filename in files_to_check:
            if os.path.exists(filename):
                size_kb = os.path.getsize(filename) / 1024
                print(f"   ✓ {filename} ({size_kb:.1f} KB)")
        
        print(f"\n📖 Open 'CHAPTER_5_RESULTS.md' to view complete results")
        print(f"⏱️  End Time: {datetime.now().strftime('%H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Chrome browser is installed")
        print("2. Check internet connection")
        print("3. Install required packages: pip install selenium pandas numpy matplotlib seaborn")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'selenium', 'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        return
    
    print("✅ All dependencies available")
    
    # Run the test
    success = run_quick_test()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Review CHAPTER_5_RESULTS.md for complete analysis")
        print("2. Check performance_analysis.png for visualizations")
        print("3. Use results in your project documentation")
    else:
        print("\n🔧 If issues persist:")
        print("1. Run 'python comparison_test.py' manually")
        print("2. Then run 'python analyze_pkl_results.py'")

if __name__ == "__main__":
    main()