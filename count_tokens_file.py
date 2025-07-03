#!/usr/bin/env python3
"""
Token Counter for Files

A professional utility for counting tokens in text files using OpenAI's tiktoken library.
Supports various file types and provides detailed analysis with robust error handling.

Author: Your Name
Version: 1.1.0
License: MIT
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import tiktoken
except ImportError:
    print("❌ Error: tiktoken library not found.")
    print("📦 Install it using: pip install tiktoken")
    sys.exit(1)


class TokenCounter:
    """A robust token counter for text files using tiktoken."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token counter with specified encoding.
        
        Args:
            encoding_name: The tokenizer encoding to use (default: cl100k_base)
        """
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {e}")
    
    def count_tokens_in_file(self, filepath: str) -> Tuple[int, dict]:
        """
        Count tokens in a given text file with detailed statistics.
        
        Args:
            filepath: Path to the input file
            
        Returns:
            Tuple of (token_count, file_stats)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnicodeDecodeError: If the file can't be decoded as UTF-8
            PermissionError: If the file can't be read due to permissions
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")
        
        # Check if file is likely binary
        if self._is_binary_file(file_path):
            raise ValueError(f"File appears to be binary: {filepath}")
        
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError as e:
            # Fixed: Use ValueError instead of UnicodeDecodeError for custom message
            raise ValueError(
                f"Unable to decode file as UTF-8: {filepath}. "
                f"Please check the file encoding. Original error: {e}"
            ) from e
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {filepath}")
        
        # Count tokens
        tokens = self.encoder.encode(content)
        token_count = len(tokens)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        file_size = file_path.stat().st_size
        
        stats = {
            'file_size_bytes': file_size,
            'file_size_kb': file_size / 1024,
            'character_count': len(content),
            'line_count': content.count('\n') + 1 if content else 0,
            'processing_time': processing_time,
            'tokens_per_character': token_count / len(content) if content else 0,
            'encoding': self.encoding_name
        }
        
        return token_count, stats
    
    @staticmethod
    def _is_binary_file(file_path: Path, chunk_size: int = 1024) -> bool:
        """
        Check if a file is likely binary by examining the first chunk.
        
        Args:
            file_path: Path to the file to check
            chunk_size: Number of bytes to read for detection
            
        Returns:
            True if file appears to be binary, False otherwise
        """
        try:
            with open(file_path, 'rb') as file:
                chunk = file.read(chunk_size)
                # Check for null bytes (common in binary files)
                if b'\x00' in chunk:
                    return True
                # Check for high ratio of non-printable characters
                printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
                if chunk and (printable_chars / len(chunk)) < 0.7:
                    return True
        except (IOError, OSError):
            return True  # Assume binary if can't read
        
        return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)  # Convert to float for division
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def print_detailed_results(filepath: str, token_count: int, stats: dict, quiet: bool = False):
    """Print formatted results with optional detailed statistics."""
    if quiet:
        print(token_count)
        return
    
    print(f"\n📄 File Analysis Results")
    print(f"{'='*50}")
    print(f"📁 File: {filepath}")
    print(f"🎯 Token Count: {token_count:,} tokens")
    
    if stats.get('show_detailed', False):
        print(f"📊 File Size: {format_file_size(stats['file_size_bytes'])}")
        print(f"📝 Characters: {stats['character_count']:,}")
        print(f"📋 Lines: {stats['line_count']:,}")
        print(f"🔤 Tokens/Char Ratio: {stats['tokens_per_character']:.3f}")
        print(f"⚙️  Encoding: {stats['encoding']}")
        print(f"⏱️  Processing Time: {stats['processing_time']:.3f} seconds")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Count tokens in a text file using tiktoken tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_tokens_file.py document.txt
  python count_tokens_file.py --quiet script.py
  python count_tokens_file.py --stats README.md
        """
    )
    
    parser.add_argument(
        'filepath',
        help='Path to the text file to analyze'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Output only the token count (useful for scripting)'
    )
    
    parser.add_argument(
        '-s', '--stats',
        action='store_true',
        help='Show detailed file statistics'
    )
    
    parser.add_argument(
        '-e', '--encoding',
        default='cl100k_base',
        help='Tokenizer encoding to use (default: cl100k_base)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Token Counter v1.1.0'
    )
    
    return parser


def main():
    """Main function with comprehensive error handling and user-friendly output."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize token counter
        counter = TokenCounter(args.encoding)
        
        # Count tokens and get statistics
        token_count, stats = counter.count_tokens_in_file(args.filepath)
        
        # Add detailed stats flag
        stats['show_detailed'] = args.stats
        
        # Display results
        print_detailed_results(args.filepath, token_count, stats, args.quiet)
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {args.filepath}")
        print("💡 Please check the file path and try again.")
        sys.exit(1)
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        if "binary" in str(e).lower():
            print("💡 This tool only works with text files.")
        elif "decode" in str(e).lower() or "encoding" in str(e).lower():
            print("💡 Please ensure the file is UTF-8 encoded or try a different encoding.")
        sys.exit(1)
    
    except PermissionError:
        print(f"❌ Error: Permission denied - {args.filepath}")
        print("💡 Please check file permissions and try again.")
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