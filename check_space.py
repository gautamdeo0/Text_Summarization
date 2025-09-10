import os
import shutil
import sys

def get_disk_space(path):
    """Get disk space information for the given path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total': total // (2**30),  # Convert to GB
            'used': used // (2**30),
            'free': free // (2**30)
        }
    except Exception as e:
        print(f"Error checking disk space: {str(e)}")
        return None

def check_space_requirements():
    """Check if there's enough space for the models"""
    # Approximate sizes of models in GB
    model_sizes = {
        'mT5_multilingual_XLSum': 1.2,
        'mbart-large-50': 2.4,
        'bert-base-multilingual-cased': 0.7
    }
    
    total_needed = sum(model_sizes.values())
    
    # Check D drive space
    drive_path = "D:\\"
    space_info = get_disk_space(drive_path)
    
    if not space_info:
        print("Could not check disk space. Please ensure D: drive is available.")
        return False
    
    print("\n=== Disk Space Information ===")
    print(f"D: Drive Total Space: {space_info['total']} GB")
    print(f"D: Drive Used Space: {space_info['used']} GB")
    print(f"D: Drive Free Space: {space_info['free']} GB")
    
    print("\n=== Model Space Requirements ===")
    for model, size in model_sizes.items():
        print(f"{model}: {size:.1f} GB")
    print(f"Total space needed: {total_needed:.1f} GB")
    
    if space_info['free'] < total_needed:
        print(f"\n⚠️ WARNING: Not enough free space on D: drive!")
        print(f"You need at least {total_needed:.1f} GB, but only have {space_info['free']} GB free.")
        return False
    else:
        print(f"\n✅ Sufficient space available on D: drive!")
        print(f"You have {space_info['free']} GB free, and need {total_needed:.1f} GB.")
        return True

if __name__ == "__main__":
    # Create the models directory if it doesn't exist
    models_dir = "D:/huggingface_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Check space requirements
    if check_space_requirements():
        print("\nYou can proceed with running the application.")
        sys.exit(0)
    else:
        print("\nPlease free up some space on D: drive before proceeding.")
        sys.exit(1) 