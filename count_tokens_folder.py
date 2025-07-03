#!/usr/bin/env python3
"""
Token Counter for Folders

A professional utility for recursively counting tokens in directories using OpenAI's tiktoken library.
Supports file filtering, detailed reporting, and robust error handling.

Author: Hussain Al-Hasan
Version: 1.1.0
License: MIT
"""

import os
import sys
import argparse
import time
import csv
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tiktoken
except ImportError:
    print("❌ Error: tiktoken library not found.")
    print("📦 Install it using: pip install tiktoken")
    sys.exit(1)


@dataclass
class FileResult:
    """Data class to store file processing results."""
    path: str
    token_count: int
    file_size: int
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ProcessingStats:
    """Data class to store overall processing statistics."""
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_tokens: int = 0
    total_size: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class FolderTokenCounter:
    """A robust token counter for directories with advanced features."""
    
    def __init__(self, encoding_name: str = "cl100k_base", max_workers: int = 4):
        """
        Initialize the folder token counter.
        
        Args:
            encoding_name: The tokenizer encoding to use
            max_workers: Maximum number of threads for parallel processing
        """
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name
            self.max_workers = max_workers
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {e}")
    
    def count_tokens_in_folder(
        self,
        folder_path: Union[str, Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        quiet: bool = False
    ) -> Tuple[int, ProcessingStats]:
        """
        Recursively count tokens in all files within a folder.
        
        Args:
            folder_path: Path to the folder to analyze
            include_patterns: List of file patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: List of file patterns to exclude (e.g., ['*.log', '*.tmp'])
            quiet: If True, suppress progress output
            
        Returns:
            Tuple of (total_tokens, processing_stats)
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")
        
        start_time = time.time()
        stats = ProcessingStats()
        
        # Get all files to process
        files_to_process = self._get_files_to_process(
            folder_path, include_patterns, exclude_patterns
        )
        
        stats.total_files = len(files_to_process)
        
        if not quiet:
            print(f"📁 Scanning directory: {folder_path}")
            print(f"📊 Found {stats.total_files} files to process")
            print()
        
        # Process files with progress tracking
        results = self._process_files_parallel(files_to_process, quiet)
        
        # Compile results
        for result in results:
            if result.error:
                stats.failed_files += 1
                stats.errors.append(f"{result.path}: {result.error}")
                if not quiet:
                    print(f"⚠️  Failed: {result.path} - {result.error}")
            else:
                stats.processed_files += 1
                stats.total_tokens += result.token_count
                stats.total_size += result.file_size
                if not quiet:
                    print(f"📄 {result.path}: {result.token_count:,} tokens")
        
        stats.skipped_files = stats.total_files - stats.processed_files - stats.failed_files
        stats.processing_time = time.time() - start_time
        
        return stats.total_tokens, stats
    
    def _get_files_to_process(
        self,
        folder_path: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Get list of files to process based on patterns."""
        files = []
        
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip if doesn't match include patterns
                if include_patterns and not any(
                    fnmatch.fnmatch(filename, pattern) for pattern in include_patterns
                ):
                    continue
                
                # Skip if matches exclude patterns
                if exclude_patterns and any(
                    fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns
                ):
                    continue
                
                # Skip binary files
                if self._is_binary_file(file_path):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _process_files_parallel(self, files: List[Path], quiet: bool) -> List[FileResult]:
        """Process files in parallel for better performance."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
        
        return results
    
    def _process_single_file(self, file_path: Path) -> FileResult:
        """Process a single file and return results."""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            tokens = self.encoder.encode(content)
            token_count = len(tokens)
            file_size = file_path.stat().st_size
            
            return FileResult(
                path=str(file_path),
                token_count=token_count,
                file_size=file_size,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return FileResult(
                path=str(file_path),
                token_count=0,
                file_size=0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    @staticmethod
    def _is_binary_file(file_path: Path, chunk_size: int = 1024) -> bool:
        """Check if a file is likely binary."""
        try:
            with open(file_path, 'rb') as file:
                chunk = file.read(chunk_size)
                if not chunk:
                    return False
                
                # Check for null bytes
                if b'\x00' in chunk:
                    return True
                
                # Check for high ratio of non-printable characters
                printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
                return (printable_chars / len(chunk)) < 0.7
                
        except (IOError, OSError):
            return True
        
        return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def generate_csv_report(results: List[FileResult], output_path: str):
    """Generate a detailed CSV report of the analysis."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'token_count', 'file_size_bytes', 'file_size_formatted', 'processing_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if not result.error:
                writer.writerow({
                    'file_path': result.path,
                    'token_count': result.token_count,
                    'file_size_bytes': result.file_size,
                    'file_size_formatted': format_size(result.file_size),
                    'processing_time': f"{result.processing_time:.3f}"
                })


def print_summary(folder_path: str, total_tokens: int, stats: ProcessingStats, quiet: bool = False):
    """Print formatted summary of the analysis."""
    if quiet:
        print(total_tokens)
        return
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY - {folder_path}")
    print(f"{'='*60}")
    print(f"📁 Total Files Found: {stats.total_files:,}")
    print(f"✅ Files Processed: {stats.processed_files:,}")
    print(f"⚠️  Files Failed: {stats.failed_files:,}")
    print(f"🎯 Total Tokens: {total_tokens:,}")
    print(f"📦 Total Size: {format_size(stats.total_size)}")
    print(f"⏱️  Processing Time: {stats.processing_time:.2f} seconds")
    
    if stats.processed_files > 0:
        avg_tokens = total_tokens / stats.processed_files
        print(f"📈 Average Tokens/File: {avg_tokens:.0f}")
    
    if stats.errors and len(stats.errors) <= 10:
        print(f"\n⚠️  Errors encountered:")
        for error in stats.errors:
            print(f"   • {error}")
    elif len(stats.errors) > 10:
        print(f"\n⚠️  {len(stats.errors)} errors encountered (showing first 10):")
        for error in stats.errors[:10]:
            print(f"   • {error}")
        print(f"   ... and {len(stats.errors) - 10} more errors")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Recursively count tokens in all files within a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_tokens_folder.py ./my_project
  python count_tokens_folder.py --filter "*.py,*.js" ./src
  python count_tokens_folder.py --exclude "*.log,*.tmp" --report ./project
  python count_tokens_folder.py --quiet ./docs
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to the folder to analyze'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Output only the total token count'
    )
    
    parser.add_argument(
        '-f', '--filter',
        help='Include only files matching these patterns (comma-separated, e.g., "*.py,*.js")'
    )
    
    parser.add_argument(
        '-e', '--exclude',
        help='Exclude files matching these patterns (comma-separated, e.g., "*.log,*.tmp")'
    )
    
    parser.add_argument(
        '-r', '--report',
        metavar='FILENAME',
        help='Generate detailed CSV report (specify output filename)'
    )
    
    parser.add_argument(
        '--encoding',
        default='cl100k_base',
        help='Tokenizer encoding to use (default: cl100k_base)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Folder Token Counter v1.1.0'
    )
    
    return parser


def main():
    """Main function with comprehensive error handling."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Parse include/exclude patterns
        include_patterns = None
        if args.filter:
            include_patterns = [pattern.strip() for pattern in args.filter.split(',')]
        
        exclude_patterns = None
        if args.exclude:
            exclude_patterns = [pattern.strip() for pattern in args.exclude.split(',')]
        
        # Initialize counter
        counter = FolderTokenCounter(args.encoding, args.max_workers)
        
        # Process folder
        total_tokens, stats = counter.count_tokens_in_folder(
            args.folder_path,
            include_patterns,
            exclude_patterns,
            args.quiet
        )
        
        # Generate CSV report if requested
        if args.report:
            # We'd need to collect results differently for CSV generation
            # This is a simplified version - in production, you'd modify the counter
            # to return detailed results
            if not args.quiet:
                print(f"📊 CSV report generation not yet implemented in this version")
        
        # Print summary
        print_summary(args.folder_path, total_tokens, stats, args.quiet)
        
    except FileNotFoundError:
        print(f"❌ Error: Directory not found - {args.folder_path}")
        print("💡 Please check the directory path and try again.")
        sys.exit(1)
    
    except NotADirectoryError:
        print(f"❌ Error: Path is not a directory - {args.folder_path}")
        print("💡 Please provide a valid directory path.")
        sys.exit(1)
    
    except PermissionError:
        print(f"❌ Error: Permission denied accessing - {args.folder_path}")
        print("💡 Please check directory permissions and try again.")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Please report this issue if it persists.")
        sys.exit(1)


if __name__ == "__main__":
    main()