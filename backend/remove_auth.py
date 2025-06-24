#!/usr/bin/env python3
"""
Script to remove authentication requirements from all endpoints for testing
"""
import re
import os

def remove_auth_from_file(filename):
    """Remove auth requirements from a Python file"""
    print(f"🔧 Processing {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace get_current_user with get_optional_user
        content = re.sub(
            r'current_user: User = Depends\(get_current_user\)',
            'current_user: Optional[User] = Depends(get_optional_user)',
            content
        )
        
        # Add get_optional_user to imports if get_current_user is imported
        if 'from auth import get_current_user' in content and 'get_optional_user' not in content:
            content = content.replace(
                'from auth import get_current_user',
                'from auth import get_current_user, get_optional_user'
            )
        
        # Add Optional import if not present and we're using Optional[User]
        if 'Optional[User]' in content and 'from typing import' in content:
            # Check if Optional is already imported
            typing_import_match = re.search(r'from typing import ([^#\n]+)', content)
            if typing_import_match:
                imports = typing_import_match.group(1)
                if 'Optional' not in imports:
                    new_imports = imports.strip() + ', Optional'
                    content = content.replace(
                        f'from typing import {imports}',
                        f'from typing import {new_imports}'
                    )
        elif 'Optional[User]' in content and 'from typing import' not in content:
            # Add typing import
            content = 'from typing import Optional\n' + content
        
        # Save if changed
        if content != original_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {filename}")
            return True
        else:
            print(f"⏩ No changes needed in {filename}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False

def main():
    """Remove auth from all relevant files"""
    files_to_process = [
        'project_management.py',
        'review_system.py', 
        'export.py',
        'file_handler.py',
        'advanced_export.py',
        'active_learning.py',
        'analytics_dashboard.py',
        'annotation_quality_dashboard.py'
    ]
    
    updated_files = []
    
    for filename in files_to_process:
        if os.path.exists(filename):
            if remove_auth_from_file(filename):
                updated_files.append(filename)
        else:
            print(f"⚠️  File not found: {filename}")
    
    print(f"\n📊 Summary:")
    print(f"Files updated: {len(updated_files)}")
    if updated_files:
        print("Updated files:")
        for f in updated_files:
            print(f"  - {f}")
    
    print("\n🎉 Authentication removed from all testing endpoints!")
    print("You can now test all Phase 1-3 and advanced features without auth!")

if __name__ == "__main__":
    main() 