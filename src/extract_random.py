# pyright: reportMissingModuleSource=false
import os
from pathlib import Path
from libzim.reader import Archive

# Use a RAW string for the Windows Network Path
ZIM_PATH = r"W:\wiki_en_all_maxi_2026-02.zim"
OUTPUT_DIR = "./wiki_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Connecting to 4TB HDD over Samba...")
zim_path = Path(ZIM_PATH)
archive = None

try:
    # 1. Open the archive
    archive = Archive(zim_path)
    print("Archive Opened. Locating Main Entry...")
    
    # 2. Grab the main entry (Wikipedia Home)
    entry = archive.main_entry
    item = entry.get_item()
    
    # 3. Read the data into a byte buffer
    # This is more stable than .content for 115GB files over a network
    content = item.content

    # 4. Clean the title for Windows
    safe_title = "".join([c for c in entry.title if c.isalnum() or c in (' ', '-', '_')])
    if not safe_title:
        safe_title = "wikipedia_home"

    output_path = os.path.join(OUTPUT_DIR, f"{safe_title}.html")
    
    # 5. Write the file locally to your PC
    with open(output_path, "wb") as f:
        f.write(content)

    print(f"Success! Saved: {entry.title} to {output_path}")

except RuntimeError as error:
    print(f"\n--- READ ERROR ---")
    print(f"The router couldn't fetch the data fast enough.")
    print(f"Error: {error}")
except Exception as e:
    print(f"Unexpected Error: {e}")
finally:
    # Manually clean up the archive handle if it was opened
    if archive:
        del archive
        print("Archive handle closed.")