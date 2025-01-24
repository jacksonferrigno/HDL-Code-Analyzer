from typing import List, Optional, Dict
from dataclasses import dataclass
import re

@dataclass
class CodeSegment:
    segment_type: str
    content: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    parent: Optional[str] = None
    semantic_tags: Optional[List[str]] = None

class VHDLSegmenter:
    def __init__(self):
        """Initialize patterns for top-level and nested parsing."""
        self.top_level_patterns = {
            'library': r'(?i)library\s+\w+;',
            'use': r'(?i)use\s+[\w\.]+;',
            'entity': r'(?i)entity\s+(\w+)\s+is.*?end\s+entity\s+\w*;',
            'architecture': r'(?i)architecture\s+(\w+)\s+of\s+(\w+)\s+is.*?end\s+architecture\s+\w*;',
            'process': r'(?i)(?:(\w+)\s*:\s*)?process\s*\(?.*?\)?.*?begin.*?end\s+process\s*\w*;',
            'package': r'(?i)package\s+(\w+)\s+is.*?end\s+package\s+\w*;',
            'component': r'(?i)component\s+(\w+)\s+is(?:\s+generic\s*\(.*?\);)?(?:\s+port\s*\(.*?\);)?.*?end\s+component\s*;',
        }
        self.nested_patterns = {
            'signal': r'(?i)signal\s+([\w,\s]+)\s*:\s*[^;]+;',
            'constant': r'(?i)constant\s+([\w,\s]+)\s*:\s*[^;]+;',
            'port_map': r'(?i)\w+\s*:\s*\w+\s+port\s+map\s*\(([^;]*?)\)\s*;',
        }
        
    def _clean_content(self, content: str) -> str:
        """Remove comments and normalize whitespace."""
        content = re.sub(r'--.*', '', content)  # Remove single-line comments
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        return content

    def _find_segments(self, content: str, patterns: Dict[str, str]) -> List[CodeSegment]:
        """Find segments based on provided patterns."""
        segments = []
        for seg_type, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.DOTALL):
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + match.group(0).count('\n')
                name = match.group(1) if match.groups() else None
                segments.append(CodeSegment(seg_type, match.group(0), start_line, end_line, name))
        return segments
    
    def _add_semantic_tags(self, segment: CodeSegment) -> List[str]:
        """Classify segments with semantic tags based on their type and content."""
        tags = []
        if segment.segment_type == 'process':
            tags.append('sequential_logic')
            if 'if' in segment.content.lower() or 'case' in segment.content.lower():
                tags.append('conditional')
            if 'wait' in segment.content.lower():
                tags.append('timing_control')
        elif segment.segment_type == 'component':
            tags.append('modular_design')
            if 'port' in segment.content.lower():
                tags.append('interface')
        elif segment.segment_type == 'entity':
            tags.append('interface_definition')
        elif segment.segment_type == 'architecture':
            tags.append('implementation')
        elif segment.segment_type in ['library', 'use']:
            tags.append('external_dependency')
        return tags

    def segment_code(self, code: str) -> List[CodeSegment]:
        """Segment VHDL code into top-level and nested constructs, with semantic tagging."""
        code = self._clean_content(code)
        top_segments = self._find_segments(code, self.top_level_patterns)
        all_segments = top_segments.copy()

        # Extract nested constructs within architectures
        for segment in top_segments:
            if segment.segment_type == 'architecture':
                nested_segments = self._find_segments(segment.content, self.nested_patterns)
                for nested in nested_segments:
                    nested.parent = segment.name
                all_segments.extend(nested_segments)

        # Add semantic tags to each segment
        for segment in all_segments:
            segment.semantic_tags = self._add_semantic_tags(segment)
        
        return sorted(all_segments, key=lambda x: x.start_line)
