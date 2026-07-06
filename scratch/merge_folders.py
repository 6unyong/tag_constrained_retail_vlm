import os
import shutil

src_dir = r"C:\Users\user\Desktop\KCL\dissertation\antigravity\data\raw\train\train"
dst_dir = r"C:\Users\user\Desktop\KCL\dissertation\antigravity\data\raw\train"

if os.path.exists(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            # Calculate relative path
            rel_path = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel_path)
            
            # Create destination dir if not exists
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            
            # Move file
            if not os.path.exists(dst_file):
                shutil.move(src_file, dst_file)
            else:
                # If it already exists, just remove the source file
                os.remove(src_file)

    # Clean up empty directories in src_dir
    shutil.rmtree(src_dir)
    print("Merge complete!")
else:
    print("Source directory does not exist.")
