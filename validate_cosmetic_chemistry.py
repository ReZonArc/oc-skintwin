#!/usr/bin/env python3
"""
Validation script for cosmetic chemistry implementation
Checks syntax, completeness, and structure of the cosmetic chemistry framework
"""

import os
import re
from pathlib import Path

def validate_atom_types_script():
   """Validate the atom_types.script file"""
   script_path = "/home/runner/work/oc-skintwin/oc-skintwin/cheminformatics/cheminformatics/types/atom_types.script"
   
   print("=== Validating atom_types.script ===")
   
   with open(script_path, 'r') as f:
       content = f.read()
   
   # Count cosmetic-specific atom types
   cosmetic_types = []
   for line in content.split('\n'):
       line = line.strip()
       if line and not line.startswith('//') and not line.startswith('#'):
           if any(keyword in line for keyword in ['INGREDIENT', 'FORMULATION', 'PROPERTY', '_LINK', 'ALLERGEN', 'SAFETY', 'REGULATORY']):
               match = re.match(r'^([A-Z_][A-Z_]*)', line)
               if match:
                   cosmetic_types.append(match.group(1))
   
   print(f"✓ Found {len(cosmetic_types)} cosmetic-specific atom types")
   
   # Check for required categories
   required_categories = [
       'ACTIVE_INGREDIENT', 'PRESERVATIVE', 'EMULSIFIER', 'HUMECTANT', 
       'SURFACTANT', 'THICKENER', 'EMOLLIENT', 'ANTIOXIDANT', 'UV_FILTER',
       'FRAGRANCE', 'COLORANT', 'PH_ADJUSTER'
   ]
   
   missing_categories = []
   for category in required_categories:
       if category not in cosmetic_types:
           missing_categories.append(category)
   
   if missing_categories:
       print(f"✗ Missing required categories: {missing_categories}")
       return False
   else:
       print("✓ All required ingredient categories present")
   
   # Check for formulation types
   formulation_types = [t for t in cosmetic_types if 'FORMULATION' in t]
   print(f"✓ Found {len(formulation_types)} formulation types")
   
   # Check for property types
   property_types = [t for t in cosmetic_types if 'PROPERTY' in t]
   print(f"✓ Found {len(property_types)} property types")
   
   # Check for interaction types
   interaction_types = [t for t in cosmetic_types if '_LINK' in t]
   print(f"✓ Found {len(interaction_types)} interaction types")
   
   return True

def validate_documentation():
   """Validate the documentation completeness"""
   doc_path = "/home/runner/work/oc-skintwin/oc-skintwin/docs/COSMETIC_CHEMISTRY.md"
   
   print("\n=== Validating Documentation ===")
   
   with open(doc_path, 'r') as f:
       content = f.read()
   
   # Check for required sections
   required_sections = [
       "Atom Type Reference",
       "Common Cosmetic Ingredients", 
       "Formulation Guidelines",
       "Regulatory Compliance",
       "Advanced Applications",
       "Usage Examples"
   ]
   
   missing_sections = []
   for section in required_sections:
       if section not in content:
           missing_sections.append(section)
   
   if missing_sections:
       print(f"✗ Missing required sections: {missing_sections}")
       return False
   else:
       print("✓ All required documentation sections present")
   
   # Check for code examples
   python_examples = content.count('```python')
   scheme_examples = content.count('```scheme')
   
   print(f"✓ Found {python_examples} Python examples in documentation")
   print(f"✓ Found {scheme_examples} Scheme examples in documentation")
   
   return True

def validate_examples():
   """Validate the example files"""
   print("\n=== Validating Examples ===")
   
   examples_dir = "/home/runner/work/oc-skintwin/oc-skintwin/cheminformatics/examples"
   
   # Check Python examples
   python_dir = os.path.join(examples_dir, "python")
   python_files = []
   if os.path.exists(python_dir):
       python_files = [f for f in os.listdir(python_dir) if f.startswith('cosmetic_') and f.endswith('.py')]
   
   print(f"✓ Found {len(python_files)} Python cosmetic examples")
   for f in python_files:
       print(f"  - {f}")
   
   # Check Scheme examples
   scheme_dir = os.path.join(examples_dir, "scheme")
   scheme_files = []
   if os.path.exists(scheme_dir):
       scheme_files = [f for f in os.listdir(scheme_dir) if f.startswith('cosmetic_') and f.endswith('.scm')]
   
   print(f"✓ Found {len(scheme_files)} Scheme cosmetic examples")
   for f in scheme_files:
       print(f"  - {f}")
   
   # Validate syntax of example files
   for python_file in python_files:
       file_path = os.path.join(python_dir, python_file)
       try:
           with open(file_path, 'r') as f:
               content = f.read()
           # Basic syntax check
           compile(content, file_path, 'exec')
           print(f"✓ {python_file} has valid Python syntax")
       except SyntaxError as e:
           print(f"✗ {python_file} has syntax error: {e}")
           return False
   
   return True

def main():
   """Main validation function"""
   print("Cosmetic Chemistry Implementation Validation")
   print("=" * 50)
   
   all_valid = True
   
   # Validate atom types
   if not validate_atom_types_script():
       all_valid = False
   
   # Validate documentation
   if not validate_documentation():
       all_valid = False
   
   # Validate examples
   if not validate_examples():
       all_valid = False
   
   print("\n" + "=" * 50)
   if all_valid:
       print("✅ All validations passed! Cosmetic chemistry implementation is complete.")
   else:
       print("❌ Some validations failed. Implementation needs improvements.")
   
   return all_valid

if __name__ == "__main__":
   main()
