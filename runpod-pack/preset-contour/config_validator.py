#!/usr/bin/env python3
"""
Config validator for ADetailer + ControlNet configuration
"""

import yaml
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path


class ConfigValidator:
    """Валідатор конфігурації для ADetailer + ControlNet"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.errors = []
        self.warnings = []
    
    def load_config(self) -> bool:
        """Завантаження конфігурації з файлу"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return True
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return False
    
    def validate_required_sections(self) -> bool:
        """Перевірка обов'язкових секцій"""
        required_sections = ['general', 'b_pass', 'io']
        missing_sections = []
        
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            self.errors.append(f"Missing required sections: {', '.join(missing_sections)}")
            return False
        
        return True
    
    def validate_general_section(self) -> bool:
        """Валідація секції general"""
        general = self.config.get('general', {})
        
        # Перевірка endpoint
        endpoint = general.get('a1111_endpoint', '')
        if not endpoint.startswith(('http://', 'https://')):
            self.warnings.append("a1111_endpoint should start with http:// or https://")
        
        # Перевірка retry параметрів
        max_retries = general.get('max_retries', 3)
        if not isinstance(max_retries, int) or max_retries < 1:
            self.errors.append("max_retries must be a positive integer")
        
        retry_delay = general.get('retry_delay', 2)
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            self.errors.append("retry_delay must be a non-negative number")
        
        return len([e for e in self.errors if 'general' in e]) == 0
    
    def validate_b_pass_section(self) -> bool:
        """Валідація секції b_pass"""
        b_pass = self.config.get('b_pass', {})
        
        # Перевірка числових параметрів
        numeric_params = {
            'denoise': (0.0, 1.0),
            'cfg': (1.0, 20.0),
            'steps': (1, 100),
            'mask_blur_px': (0, 50)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            value = b_pass.get(param)
            if value is not None:
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    self.errors.append(f"b_pass.{param} must be between {min_val} and {max_val}")
        
        # Перевірка ControlNet параметрів
        if b_pass.get('use_controlnet'):
            self._validate_controlnet_params(b_pass, 'controlnet')
        
        if b_pass.get('use_controlnet2'):
            self._validate_controlnet_params(b_pass, 'controlnet2')
        
        # Режим послідовного запуску з ранньою зупинкою
        if 'use_sequential_adetailer' in b_pass and not isinstance(b_pass['use_sequential_adetailer'], bool):
            self.errors.append("b_pass.use_sequential_adetailer must be a boolean")
        if 'use_true_fallback_adetailer' in b_pass and not isinstance(b_pass['use_true_fallback_adetailer'], bool):
            self.errors.append("b_pass.use_true_fallback_adetailer must be a boolean")

        # Manual fallback flags
        if 'enable_manual_fallback' in b_pass and not isinstance(b_pass['enable_manual_fallback'], bool):
            self.errors.append("b_pass.enable_manual_fallback must be a boolean")
        if 'open_ui_on_manual' in b_pass and not isinstance(b_pass['open_ui_on_manual'], bool):
            self.errors.append("b_pass.open_ui_on_manual must be a boolean")

        # Перевірка ADetailer параметрів
        if b_pass.get('use_adetailer'):
            self._validate_adetailer_params(b_pass, 'ad')
        
        if b_pass.get('use_adetailer2'):
            self._validate_adetailer_params(b_pass, 'ad2')
        
        # Optional third ADetailer pass
        if b_pass.get('use_adetailer3'):
            self._validate_adetailer_params(b_pass, 'ad3')
        
        return len([e for e in self.errors if 'b_pass' in e]) == 0
    
    def _validate_controlnet_params(self, b_pass: Dict, prefix: str) -> None:
        """Валідація параметрів ControlNet"""
        weight = b_pass.get(f'{prefix}_weight')
        if weight is not None and (not isinstance(weight, (int, float)) or weight < 0 or weight > 2):
            self.errors.append(f"b_pass.{prefix}_weight must be between 0 and 2")
        
        guidance_start = b_pass.get(f'{prefix}_guidance_start')
        guidance_end = b_pass.get(f'{prefix}_guidance_end')
        
        if guidance_start is not None and guidance_end is not None:
            if not isinstance(guidance_start, (int, float)) or not isinstance(guidance_end, (int, float)):
                self.errors.append(f"b_pass.{prefix}_guidance_start and {prefix}_guidance_end must be numbers")
            elif guidance_start >= guidance_end:
                self.errors.append(f"b_pass.{prefix}_guidance_start must be less than {prefix}_guidance_end")
    
    def _validate_adetailer_params(self, b_pass: Dict, prefix: str) -> None:
        """Валідація параметрів ADetailer"""
        confidence = b_pass.get(f'{prefix}_confidence')
        if confidence is not None and (not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1):
            self.errors.append(f"b_pass.{prefix}_confidence must be between 0 and 1")
        
        denoise = b_pass.get(f'{prefix}_denoise')
        if denoise is not None and (not isinstance(denoise, (int, float)) or denoise < 0 or denoise > 1):
            self.errors.append(f"b_pass.{prefix}_denoise must be between 0 and 1")
    
    def validate_io_section(self) -> bool:
        """Валідація секції io"""
        io = self.config.get('io', {})
        
        # Перевірка директорій
        for dir_key in ['input_dir', 'work_dir', 'output_dir']:
            dir_path = io.get(dir_key)
            if dir_path:
                # Перевірка, чи можна створити директорію
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.warnings.append(f"Cannot create directory {dir_key}: {dir_path}")
        
        # Перевірка форматів файлів
        supported_formats = io.get('supported_formats', [])
        if not isinstance(supported_formats, list):
            self.errors.append("io.supported_formats must be a list")
        else:
            valid_formats = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff']
            for fmt in supported_formats:
                if fmt.lower() not in valid_formats:
                    self.warnings.append(f"Unsupported format: {fmt}")
        
        return len([e for e in self.errors if 'io' in e]) == 0
    
    def validate_quality_section(self) -> bool:
        """Валідація секції quality (якщо є)"""
        quality = self.config.get('quality', {})
        if not quality:
            return True
        
        # Перевірка параметрів якості
        if 'enable_quality_check' in quality and not isinstance(quality['enable_quality_check'], bool):
            self.errors.append("quality.enable_quality_check must be a boolean")
        
        # Перевірка порогових значень
        thresholds = {
            'min_face_detection_confidence': (0.0, 1.0),
            'max_blur_threshold': (0, 1000),
            'min_sharpness_score': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in thresholds.items():
            value = quality.get(param)
            if value is not None:
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    self.errors.append(f"quality.{param} must be between {min_val} and {max_val}")
        
        return len([e for e in self.errors if 'quality' in e]) == 0
    
    def validate_logging_section(self) -> bool:
        """Валідація секції logging (якщо є)"""
        logging_config = self.config.get('logging', {})
        if not logging_config:
            return True
        
        # Перевірка рівня логування
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level = logging_config.get('level', 'INFO')
        if level not in valid_levels:
            self.errors.append(f"logging.level must be one of: {', '.join(valid_levels)}")
        
        # Перевірка файлу логування
        if logging_config.get('file_logging'):
            log_file = logging_config.get('log_file')
            if log_file:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.warnings.append(f"Cannot create log file directory: {e}")
        
        return len([e for e in self.errors if 'logging' in e]) == 0
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Повна валідація конфігурації"""
        if not self.load_config():
            return False, self.errors, self.warnings
        
        validators = [
            self.validate_required_sections,
            self.validate_general_section,
            self.validate_b_pass_section,
            self.validate_io_section,
            self.validate_quality_section,
            self.validate_logging_section
        ]
        
        for validator in validators:
            try:
                validator()
            except Exception as e:
                self.errors.append(f"Validation error in {validator.__name__}: {e}")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def print_report(self) -> None:
        """Виведення звіту про валідацію"""
        print(f"\n=== Config Validation Report ===")
        print(f"Config file: {self.config_path}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid!")
        elif not self.errors:
            print("\n✅ Configuration is valid with warnings")
        else:
            print("\n❌ Configuration has errors")


def main():
    """Головна функція для CLI використання"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Validate ADetailer + ControlNet configuration")
    ap.add_argument("config", help="Path to config file")
    args = ap.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    validator = ConfigValidator(args.config)
    is_valid, errors, warnings = validator.validate_all()
    
    validator.print_report()
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
