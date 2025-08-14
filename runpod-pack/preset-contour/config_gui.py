#!/usr/bin/env python3
"""
Simple GUI for editing ADetailer + ControlNet configuration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigEditorGUI:
    """Графічний інтерфейс для редагування конфігурації"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ADetailer + ControlNet Config Editor")
        self.root.geometry("800x600")
        
        self.config_data = {}
        self.config_path = ""
        
        self.setup_ui()
        self.load_default_config()
    
    def setup_ui(self):
        """Налаштування інтерфейсу"""
        # Головне меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Config", command=self.open_config)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_command(label="Save As...", command=self.save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Головний контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Налаштування grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Кнопки
        ttk.Button(main_frame, text="Open Config", command=self.open_config).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Button(main_frame, text="Save Config", command=self.save_config).grid(row=0, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Button(main_frame, text="Validate Config", command=self.validate_config).grid(row=0, column=2, sticky=tk.W, pady=(0, 10))
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        main_frame.rowconfigure(1, weight=1)
        
        # Створення вкладок
        self.create_general_tab()
        self.create_b_pass_tab()
        self.create_io_tab()
        self.create_quality_tab()
        self.create_logging_tab()
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def create_general_tab(self):
        """Створення вкладки General"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="General")
        
        # Backend
        ttk.Label(frame, text="Backend:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.backend_var = tk.StringVar(value="a1111")
        backend_combo = ttk.Combobox(frame, textvariable=self.backend_var, values=["a1111", "comfyui"], state="readonly")
        backend_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Endpoint
        ttk.Label(frame, text="A1111 Endpoint:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.endpoint_var = tk.StringVar(value="http://127.0.0.1:7860")
        ttk.Entry(frame, textvariable=self.endpoint_var, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Max retries
        ttk.Label(frame, text="Max Retries:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_retries_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.max_retries_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Retry delay
        ttk.Label(frame, text="Retry Delay (s):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.retry_delay_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.retry_delay_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_b_pass_tab(self):
        """Створення вкладки B-Pass"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="B-Pass")
        
        # Prompt
        ttk.Label(frame, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.prompt_var = tk.StringVar()
        prompt_text = tk.Text(frame, height=4, width=60)
        prompt_text.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.prompt_text = prompt_text
        
        # Negative prompt
        ttk.Label(frame, text="Negative Prompt:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.negative_var = tk.StringVar()
        negative_text = tk.Text(frame, height=3, width=60)
        negative_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.negative_text = negative_text
        
        # Основні параметри
        ttk.Label(frame, text="Denoise:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.denoise_var = tk.DoubleVar(value=0.18)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.denoise_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(frame, text="CFG:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.cfg_var = tk.DoubleVar(value=5.0)
        ttk.Scale(frame, from_=1.0, to=20.0, variable=self.cfg_var, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(frame, text="Steps:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.steps_var = tk.IntVar(value=32)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.steps_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_io_tab(self):
        """Створення вкладки I/O"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="I/O")
        
        # Input directory
        ttk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_dir_var = tk.StringVar(value="input")
        ttk.Entry(frame, textvariable=self.input_dir_var, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.input_dir_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Work directory
        ttk.Label(frame, text="Work Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.work_dir_var = tk.StringVar(value="work")
        ttk.Entry(frame, textvariable=self.work_dir_var, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.work_dir_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(frame, textvariable=self.output_dir_var, width=40).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.output_dir_var)).grid(row=2, column=2, padx=5, pady=5)
        
        # Backup original
        self.backup_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Backup Original Images", variable=self.backup_original_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_quality_tab(self):
        """Створення вкладки Quality"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Quality")
        
        # Enable quality check
        self.enable_quality_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Enable Quality Check", variable=self.enable_quality_check_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Face detection confidence
        ttk.Label(frame, text="Min Face Detection Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.face_confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.face_confidence_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Blur threshold
        ttk.Label(frame, text="Max Blur Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.blur_threshold_var = tk.IntVar(value=100)
        ttk.Scale(frame, from_=0, to=1000, variable=self.blur_threshold_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_logging_tab(self):
        """Створення вкладки Logging"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Logging")
        
        # Log level
        ttk.Label(frame, text="Log Level:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(frame, textvariable=self.log_level_var, 
                                      values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], state="readonly")
        log_level_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # File logging
        self.file_logging_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Enable File Logging", variable=self.file_logging_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Log file
        ttk.Label(frame, text="Log File:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.log_file_var = tk.StringVar(value="adetailer_batch.log")
        ttk.Entry(frame, textvariable=self.log_file_var, width=40).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.log_file_var)).grid(row=2, column=2, padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def browse_directory(self, var: tk.StringVar):
        """Вибір директорії"""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)
    
    def browse_file(self, var: tk.StringVar):
        """Вибір файлу"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def load_default_config(self):
        """Завантаження конфігурації за замовчуванням"""
        self.config_data = {
            'general': {
                'backend': 'a1111',
                'a1111_endpoint': 'http://127.0.0.1:7860',
                'max_retries': 3,
                'retry_delay': 2.0
            },
            'b_pass': {
                'prompt': 'ultra sharp contouring, very strong cheekbone definition...',
                'negative': 'waxy skin, plastic look, smeared makeup...',
                'denoise': 0.18,
                'cfg': 5.0,
                'steps': 32
            },
            'io': {
                'input_dir': 'input',
                'work_dir': 'work',
                'output_dir': 'output',
                'backup_original': True
            },
            'quality': {
                'enable_quality_check': True,
                'min_face_detection_confidence': 0.5,
                'max_blur_threshold': 100
            },
            'logging': {
                'level': 'INFO',
                'file_logging': True,
                'log_file': 'adetailer_batch.log'
            }
        }
        self.update_ui_from_config()
    
    def update_ui_from_config(self):
        """Оновлення UI з конфігурації"""
        # General
        self.backend_var.set(self.config_data.get('general', {}).get('backend', 'a1111'))
        self.endpoint_var.set(self.config_data.get('general', {}).get('a1111_endpoint', 'http://127.0.0.1:7860'))
        self.max_retries_var.set(self.config_data.get('general', {}).get('max_retries', 3))
        self.retry_delay_var.set(self.config_data.get('general', {}).get('retry_delay', 2.0))
        
        # B-Pass
        self.prompt_text.delete('1.0', tk.END)
        self.prompt_text.insert('1.0', self.config_data.get('b_pass', {}).get('prompt', ''))
        self.negative_text.delete('1.0', tk.END)
        self.negative_text.insert('1.0', self.config_data.get('b_pass', {}).get('negative', ''))
        self.denoise_var.set(self.config_data.get('b_pass', {}).get('denoise', 0.18))
        self.cfg_var.set(self.config_data.get('b_pass', {}).get('cfg', 5.0))
        self.steps_var.set(self.config_data.get('b_pass', {}).get('steps', 32))
        
        # I/O
        self.input_dir_var.set(self.config_data.get('io', {}).get('input_dir', 'input'))
        self.work_dir_var.set(self.config_data.get('io', {}).get('work_dir', 'work'))
        self.output_dir_var.set(self.config_data.get('io', {}).get('output_dir', 'output'))
        self.backup_original_var.set(self.config_data.get('io', {}).get('backup_original', True))
        
        # Quality
        self.enable_quality_check_var.set(self.config_data.get('quality', {}).get('enable_quality_check', True))
        self.face_confidence_var.set(self.config_data.get('quality', {}).get('min_face_detection_confidence', 0.5))
        self.blur_threshold_var.set(self.config_data.get('quality', {}).get('max_blur_threshold', 100))
        
        # Logging
        self.log_level_var.set(self.config_data.get('logging', {}).get('level', 'INFO'))
        self.file_logging_var.set(self.config_data.get('logging', {}).get('file_logging', True))
        self.log_file_var.set(self.config_data.get('logging', {}).get('log_file', 'adetailer_batch.log'))
    
    def update_config_from_ui(self):
        """Оновлення конфігурації з UI"""
        self.config_data = {
            'general': {
                'backend': self.backend_var.get(),
                'a1111_endpoint': self.endpoint_var.get(),
                'max_retries': self.max_retries_var.get(),
                'retry_delay': self.retry_delay_var.get()
            },
            'b_pass': {
                'prompt': self.prompt_text.get('1.0', tk.END).strip(),
                'negative': self.negative_text.get('1.0', tk.END).strip(),
                'denoise': self.denoise_var.get(),
                'cfg': self.cfg_var.get(),
                'steps': self.steps_var.get()
            },
            'io': {
                'input_dir': self.input_dir_var.get(),
                'work_dir': self.work_dir_var.get(),
                'output_dir': self.output_dir_var.get(),
                'backup_original': self.backup_original_var.get()
            },
            'quality': {
                'enable_quality_check': self.enable_quality_check_var.get(),
                'min_face_detection_confidence': self.face_confidence_var.get(),
                'max_blur_threshold': self.blur_threshold_var.get()
            },
            'logging': {
                'level': self.log_level_var.get(),
                'file_logging': self.file_logging_var.get(),
                'log_file': self.log_file_var.get()
            }
        }
    
    def open_config(self):
        """Відкриття конфігурації"""
        filename = filedialog.askopenfilename(
            title="Open Config File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
                self.config_path = filename
                self.update_ui_from_config()
                self.status_var.set(f"Loaded config: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")
    
    def save_config(self):
        """Збереження конфігурації"""
        if not self.config_path:
            self.save_config_as()
        else:
            self._save_config_to_file(self.config_path)
    
    def save_config_as(self):
        """Збереження конфігурації як"""
        filename = filedialog.asksaveasfilename(
            title="Save Config File",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.config_path = filename
            self._save_config_to_file(filename)
    
    def _save_config_to_file(self, filename: str):
        """Збереження конфігурації у файл"""
        try:
            self.update_config_from_ui()
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2, allow_unicode=True)
            self.status_var.set(f"Saved config: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def validate_config(self):
        """Валідація конфігурації"""
        try:
            from config_validator import ConfigValidator
            
            if not self.config_path:
                messagebox.showwarning("Warning", "Please save config first")
                return
            
            validator = ConfigValidator(self.config_path)
            is_valid, errors, warnings = validator.validate_all()
            
            if is_valid:
                messagebox.showinfo("Validation", "Configuration is valid!")
            else:
                error_msg = "Configuration has errors:\n\n" + "\n".join(errors)
                if warnings:
                    error_msg += "\n\nWarnings:\n" + "\n".join(warnings)
                messagebox.showerror("Validation Failed", error_msg)
                
        except ImportError:
            messagebox.showerror("Error", "Config validator not found. Please install required dependencies.")
        except Exception as e:
            messagebox.showerror("Error", f"Validation failed: {e}")


def main():
    """Головна функція"""
    root = tk.Tk()
    app = ConfigEditorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
