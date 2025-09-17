#!/usr/bin/env python3
"""
프로세스4 오류 수정 패치 도구
- TopologyException 해결
- 좌표계 변환 오류 해결
- 자동 백업 및 복구 기능
"""

import os
import sys
import shutil
import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import hashlib

class Process4Patcher:
    def __init__(self):
        self.patch_version = "v1.0.0"
        self.patch_date = "2025-01-28"
        
        # 실행 파일 경로 감지
        if getattr(sys, 'frozen', False):
            # PyInstaller로 빌드된 경우
            self.exe_dir = Path(sys.executable).parent
        else:
            # 개발 환경
            self.exe_dir = Path(__file__).parent
            
        # 패치 대상 경로들
        self.internal_dir = self.exe_dir / "_internal"
        self.backup_dir = self.exe_dir / "patch_backup"
        
        # 패치할 파일들 정의
        self.patch_files = {
            "district_road_clipper.py": {
                "target": self.internal_dir / "src" / "core" / "district_road_clipper.py",
                "source": Path(__file__).parent / "src" / "core" / "district_road_clipper.py",
                "description": "TopologyException 오류 해결"
            },
            "process4_inference.py": {
                "target": self.internal_dir / "process4_inference.py", 
                "source": Path(__file__).parent / "process4_inference.py",
                "description": "좌표계 변환 오류 해결"
            }
        }
        
        # GUI 초기화
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 인터페이스 설정"""
        self.root = tk.Tk()
        self.root.title(f"프로세스4 오류 수정 패치 {self.patch_version}")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="🔧 프로세스4 오류 수정 패치", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 패치 정보
        info_frame = ttk.LabelFrame(main_frame, text="패치 정보", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(info_frame, text=f"버전: {self.patch_version}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, text=f"날짜: {self.patch_date}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="해결 문제: TopologyException, 좌표계 변환 오류").grid(row=2, column=0, sticky=tk.W)
        
        # 패치 대상 파일 목록
        files_frame = ttk.LabelFrame(main_frame, text="패치 대상 파일", padding="10")
        files_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.file_tree = ttk.Treeview(files_frame, columns=("status", "description"), show="tree headings", height=6)
        self.file_tree.heading("#0", text="파일명")
        self.file_tree.heading("status", text="상태")
        self.file_tree.heading("description", text="설명")
        self.file_tree.column("#0", width=200)
        self.file_tree.column("status", width=100)
        self.file_tree.column("description", width=250)
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        # 진행률 바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 상태 텍스트
        self.status_var = tk.StringVar(value="패치 준비 완료")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=2, pady=(0, 20))
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        # 버튼들
        self.check_btn = ttk.Button(button_frame, text="🔍 환경 검사", command=self.check_environment)
        self.check_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.patch_btn = ttk.Button(button_frame, text="🔧 패치 적용", command=self.apply_patch, state=tk.DISABLED)
        self.patch_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.rollback_btn = ttk.Button(button_frame, text="↩️ 롤백", command=self.rollback_patch, state=tk.DISABLED)
        self.rollback_btn.grid(row=0, column=2, padx=(0, 10))
        
        self.exit_btn = ttk.Button(button_frame, text="❌ 종료", command=self.root.quit)
        self.exit_btn.grid(row=0, column=3)
        
        # 로그 영역
        log_frame = ttk.LabelFrame(main_frame, text="로그", padding="10")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        
        self.log_text = tk.Text(log_frame, height=8, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # 초기 파일 목록 업데이트
        self.update_file_list()
        
    def log(self, message):
        """로그 메시지 출력"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update()
        
    def update_file_list(self):
        """파일 목록 업데이트"""
        # 기존 항목 삭제
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
            
        # 파일 상태 확인 및 추가
        for filename, info in self.patch_files.items():
            if info["target"].exists():
                status = "✅ 발견"
            else:
                status = "❌ 없음"
                
            self.file_tree.insert("", tk.END, text=filename, 
                                values=(status, info["description"]))
    
    def check_environment(self):
        """환경 검사"""
        self.log("🔍 환경 검사 시작...")
        self.progress_var.set(0)
        
        try:
            # 1. 실행 파일 디렉토리 확인
            self.log(f"실행 파일 경로: {self.exe_dir}")
            if not self.exe_dir.exists():
                raise Exception("실행 파일 디렉토리를 찾을 수 없습니다")
            self.progress_var.set(20)
            
            # 2. _internal 디렉토리 확인
            self.log(f"_internal 디렉토리: {self.internal_dir}")
            if not self.internal_dir.exists():
                raise Exception("_internal 디렉토리를 찾을 수 없습니다. PyInstaller로 빌드된 실행파일이 아닙니다.")
            self.progress_var.set(40)
            
            # 3. 패치 소스 파일 확인
            missing_sources = []
            for filename, info in self.patch_files.items():
                if not info["source"].exists():
                    missing_sources.append(filename)
            
            if missing_sources:
                raise Exception(f"패치 소스 파일이 없습니다: {', '.join(missing_sources)}")
            self.progress_var.set(60)
            
            # 4. 대상 파일 확인
            missing_targets = []
            for filename, info in self.patch_files.items():
                if not info["target"].exists():
                    missing_targets.append(filename)
            
            if missing_targets:
                self.log(f"⚠️ 대상 파일이 없습니다: {', '.join(missing_targets)}")
            self.progress_var.set(80)
            
            # 5. 백업 디렉토리 생성
            self.backup_dir.mkdir(exist_ok=True)
            self.log(f"백업 디렉토리: {self.backup_dir}")
            self.progress_var.set(100)
            
            self.log("✅ 환경 검사 완료!")
            self.status_var.set("패치 적용 준비 완료")
            self.patch_btn.config(state=tk.NORMAL)
            
            # 파일 목록 업데이트
            self.update_file_list()
            
        except Exception as e:
            self.log(f"❌ 환경 검사 실패: {str(e)}")
            self.status_var.set("환경 검사 실패")
            messagebox.showerror("오류", f"환경 검사 실패:\n{str(e)}")
    
    def apply_patch(self):
        """패치 적용"""
        # 확인 대화상자
        if not messagebox.askyesno("패치 적용", 
                                  "패치를 적용하시겠습니까?\n\n"
                                  "⚠️ 원본 파일은 자동으로 백업됩니다.\n"
                                  "⚠️ 프로세스4가 실행 중이면 먼저 종료해주세요."):
            return
            
        self.log("🔧 패치 적용 시작...")
        self.progress_var.set(0)
        
        try:
            # 프로세스4 실행 중인지 확인
            if self.is_process4_running():
                if messagebox.askyesno("프로세스 종료", 
                                     "프로세스4가 실행 중입니다.\n자동으로 종료하시겠습니까?"):
                    self.kill_process4()
                else:
                    self.log("❌ 프로세스4를 먼저 종료해주세요")
                    return
            
            total_files = len(self.patch_files)
            current_file = 0
            
            # 백업 타임스탬프
            backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.backup_dir / f"backup_{backup_timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            for filename, info in self.patch_files.items():
                current_file += 1
                progress = (current_file / total_files) * 100
                self.progress_var.set(progress)
                
                self.log(f"📁 처리 중: {filename}")
                
                # 대상 파일이 존재하면 백업
                if info["target"].exists():
                    backup_path = backup_subdir / filename
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(info["target"], backup_path)
                    self.log(f"💾 백업 완료: {backup_path}")
                
                # 대상 디렉토리 생성
                info["target"].parent.mkdir(parents=True, exist_ok=True)
                
                # 패치 파일 복사
                shutil.copy2(info["source"], info["target"])
                self.log(f"✅ 패치 적용: {info['target']}")
            
            self.progress_var.set(100)
            self.log("🎉 패치 적용 완료!")
            self.status_var.set("패치 적용 완료")
            
            # 버튼 상태 업데이트
            self.patch_btn.config(state=tk.DISABLED)
            self.rollback_btn.config(state=tk.NORMAL)
            
            # 파일 목록 업데이트
            self.update_file_list()
            
            messagebox.showinfo("완료", 
                              f"패치가 성공적으로 적용되었습니다!\n\n"
                              f"백업 위치: {backup_subdir}\n\n"
                              f"이제 프로세스4를 실행하여 오류가 해결되었는지 확인해보세요.")
            
        except Exception as e:
            self.log(f"❌ 패치 적용 실패: {str(e)}")
            self.status_var.set("패치 적용 실패")
            messagebox.showerror("오류", f"패치 적용 실패:\n{str(e)}")
    
    def rollback_patch(self):
        """패치 롤백"""
        if not messagebox.askyesno("롤백", "패치를 롤백하시겠습니까?\n원본 파일로 복구됩니다."):
            return
            
        self.log("↩️ 패치 롤백 시작...")
        
        try:
            # 가장 최근 백업 찾기
            backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")]
            if not backup_dirs:
                raise Exception("백업 파일을 찾을 수 없습니다")
            
            latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
            self.log(f"백업 디렉토리: {latest_backup}")
            
            # 프로세스4 실행 중인지 확인
            if self.is_process4_running():
                if messagebox.askyesno("프로세스 종료", 
                                     "프로세스4가 실행 중입니다.\n자동으로 종료하시겠습니까?"):
                    self.kill_process4()
                else:
                    self.log("❌ 프로세스4를 먼저 종료해주세요")
                    return
            
            # 백업 파일 복원
            for filename, info in self.patch_files.items():
                backup_file = latest_backup / filename
                if backup_file.exists():
                    shutil.copy2(backup_file, info["target"])
                    self.log(f"✅ 복원 완료: {filename}")
                else:
                    self.log(f"⚠️ 백업 파일 없음: {filename}")
            
            self.log("🎉 롤백 완료!")
            self.status_var.set("롤백 완료")
            
            # 버튼 상태 업데이트
            self.patch_btn.config(state=tk.NORMAL)
            self.rollback_btn.config(state=tk.DISABLED)
            
            messagebox.showinfo("완료", "패치가 성공적으로 롤백되었습니다!")
            
        except Exception as e:
            self.log(f"❌ 롤백 실패: {str(e)}")
            self.status_var.set("롤백 실패")
            messagebox.showerror("오류", f"롤백 실패:\n{str(e)}")
    
    def is_process4_running(self):
        """프로세스4 실행 중인지 확인"""
        try:
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Process4*.exe'], 
                                  capture_output=True, text=True)
            return 'Process4' in result.stdout
        except:
            return False
    
    def kill_process4(self):
        """프로세스4 종료"""
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'Process4*.exe'], check=False)
            self.log("🔄 프로세스4 종료 완료")
        except Exception as e:
            self.log(f"⚠️ 프로세스4 종료 실패: {e}")
    
    def run(self):
        """패치 도구 실행"""
        self.log(f"🚀 프로세스4 패치 도구 시작 ({self.patch_version})")
        self.log("먼저 '환경 검사' 버튼을 클릭하여 패치 환경을 확인하세요.")
        self.root.mainloop()

if __name__ == "__main__":
    try:
        patcher = Process4Patcher()
        patcher.run()
    except Exception as e:
        messagebox.showerror("오류", f"패치 도구 실행 오류:\n{str(e)}")
