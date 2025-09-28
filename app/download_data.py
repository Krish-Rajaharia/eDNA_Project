#!/usr/bin/env python3
"""
Data downloader for eDNA project
Downloads and prepares BLAST databases
"""

import os
import sys
import time
import urllib.request
import urllib.error # NEW: Import for handling HTTP errors
import subprocess
from pathlib import Path
import numpy as np
from collections import Counter
import tarfile # NEW: Import for handling .tar.gz files
import shutil # For checking if command exists

class DataDownloader:
    """Smart downloader for NCBI data with space constraints"""
    
    def __init__(self, data_dir="data", max_space_mb=500):
        self.data_dir = Path(data_dir)
        self.max_space_mb = max_space_mb
        # CHANGED: Removed trailing slash to prevent double slashes in URLs
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/blast/db"
        self.data_dir.mkdir(exist_ok=True)
        
    def get_available_databases(self):
        """Get list of available BLAST databases"""
        print("🔍 Checking available BLAST databases...")
        
        # Small, taxonomy-rich databases that fit in 500MB
        priority_dbs = [
            "16S_ribosomal_RNA",    # Available as individual files
            "18S_fungal_sequences", # Available as .tar.gz
            "28S_ribosomal_RNA",    # Available as .tar.gz
            "ITS_RefSeq_Fungi",     # Available as .tar.gz
        ]
        
        return priority_dbs
    
    def _check_blast_available(self):
        """Check if BLAST+ tools are available"""
        return shutil.which("blastdbcmd") is not None
    
    # =========================================================================
    # MODIFIED FUNCTION
    # =========================================================================
    def download_database(self, db_name):
        """
        Download and extract specific BLAST database.
        Handles both .tar.gz archives and individual files.
        """
        print(f"📥 Downloading {db_name}...")
        
        # --- Attempt to download the .tar.gz archive first ---
        tar_url = f"{self.base_url}/{db_name}.tar.gz"
        tar_path = self.data_dir / f"{db_name}.tar.gz"
        
        try:
            print(f"Trying to download archive: {tar_url}")
            urllib.request.urlretrieve(tar_url, tar_path)
            
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            
            # Clean up the downloaded archive
            os.remove(tar_path)
            print("Archive extracted and removed.")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # --- .tar.gz not found, fall back to individual files ---
                print(f"Archive not found. Trying individual files for {db_name}...")
                db_files = ["nhr", "nin", "nsq"]
                success = True
                
                for ext in db_files:
                    url = f"{self.base_url}/{db_name}.{ext}"
                    local_path = self.data_dir / f"{db_name}.{ext}"
                    
                    try:
                        if not local_path.exists():
                            print(f"Downloading {url}...")
                            urllib.request.urlretrieve(url, local_path)
                    except Exception as download_e:
                        print(f"❌ Failed to download {url}: {download_e}")
                        success = False
                        break
                
                if not success:
                    return False
            else:
                print(f"❌ An HTTP error occurred: {e}")
                return False

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return False

        # --- Check if BLAST+ is available before attempting conversion ---
        if not self._check_blast_available():
            print(f"⚠️  BLAST+ tools not found. Database downloaded but not converted to FASTA.")
            print(f"📁 Database files are available in: {self.data_dir}")
            print("💡 Install NCBI BLAST+ to enable FASTA conversion.")
            print("   Download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/")
            return True  # Still return True since download was successful

        # --- Convert BLAST db to FASTA ---
        print(f"📝 Converting {db_name} to FASTA format...")
        output_fasta = self.data_dir / f"{db_name}.fasta"
        
        try:
            # Use blastdbcmd on all platforms. This is the correct tool.
            blast_cmd = "blastdbcmd"
            
            cmd = f"{blast_cmd} -db {self.data_dir / db_name} -entry all -outfmt '%f' > {output_fasta}"
            # Use shell=True carefully, but it's acceptable here for redirection.
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            print(f"✅ Successfully created FASTA file: {output_fasta}")
            return True
            
        except subprocess.CalledProcessError as e:
            # This error catches when the command itself fails (e.g., blastdbcmd not found)
            print(f"❌ Failed to convert to FASTA. The command failed.")
            print(f"Error details: {e.stderr}")
            print("\n💡 Make sure NCBI BLAST+ is installed and its 'bin' directory is in your system's PATH.")
            if sys.platform == "win32":
                print("Download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/")
            else:
                print("Try: sudo apt-get install ncbi-blast+ (Debian/Ubuntu) or brew install blast (macOS)")
            return False
        except FileNotFoundError:
            # This error catches when the command isn't found at all
            print(f"❌ Command '{blast_cmd}' not found.")
            print("\n💡 Make sure NCBI BLAST+ is installed and its 'bin' directory is in your system's PATH.")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during FASTA conversion: {e}")
            return False
    
    def create_synthetic_data(self):
        """Create synthetic DNA sequences for testing"""
        print("🧪 Creating synthetic eDNA dataset...")
        
        taxa_patterns = {
            'bacteria': ['ATGC' * 10, 'GCTA' * 10, 'TACG' * 10],
            'archaea': ['CGAT' * 10, 'TAGC' * 10, 'ATCG' * 10],
            'fungi': ['GCAT' * 10, 'ACGT' * 10, 'TGCA' * 10],
            'protist': ['CATG' * 10, 'GTAC' * 10, 'AGTC' * 10],
            'plant': ['TCAG' * 10, 'GACT' * 10, 'CAGT' * 10]
        }
        
        sequences = []
        labels = []
        
        for taxon, patterns in taxa_patterns.items():
            for pattern in patterns:
                for i in range(200):
                    seq = self._mutate_sequence(pattern, mutation_rate=0.1)
                    sequences.append(seq)
                    labels.append(taxon)
        
        synthetic_file = self.data_dir / "synthetic_sequences.fasta"
        with open(synthetic_file, 'w') as f:
            for i, (seq, label) in enumerate(zip(sequences, labels)):
                f.write(f">{label}_{i}\n{seq}\n")
                
        print(f"✅ Created {len(sequences)} synthetic sequences")
        return "synthetic_sequences"
    
    def _mutate_sequence(self, sequence, mutation_rate=0.1):
        """Add random mutations to sequence"""
        bases = list(sequence)
        n_mutations = int(len(bases) * mutation_rate)
        
        for _ in range(n_mutations):
            pos = np.random.randint(len(bases))
            bases[pos] = np.random.choice(['A', 'T', 'C', 'G'])
            
        return ''.join(bases)

def main():
    """Main data downloading script"""
    downloader = DataDownloader()
    
    print("🧬 eDNA Data Downloader")
    print(f"📊 Space constraint: {downloader.max_space_mb}MB")
    
    # Check if BLAST+ is available
    if downloader._check_blast_available():
        print("✅ BLAST+ tools detected")
    else:
        print("⚠️  BLAST+ tools not found - databases will be downloaded but not converted to FASTA")
    
    available_dbs = downloader.get_available_databases()
    print("\nAvailable databases:")
    for i, db in enumerate(available_dbs, 1):
        print(f"{i}. {db}")
    
    while True:
        choice = input("\nEnter database number to download (or 's' for synthetic data, 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            break
        elif choice.lower() == 's':
            db_name = downloader.create_synthetic_data()
            print(f"\n✅ Synthetic data saved as: {db_name}.fasta")
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_dbs):
                db_name = available_dbs[idx]
                if downloader.download_database(db_name):
                    if downloader._check_blast_available():
                        print(f"\n✅ Database downloaded and converted: {db_name}.fasta")
                    else:
                        print(f"\n✅ Database downloaded (BLAST format): {db_name}")
                        print("💡 Install BLAST+ to convert to FASTA format")
                    break
                else:
                    print("\n❌ Failed to download database. Try another one or use synthetic data.")
            else:
                print("\n❌ Invalid choice. Please try again.")
        except ValueError:
            print("\n❌ Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()