#!/usr/bin/env python3
"""
프로세스4 내부망 패치 배포 도구
_internal 폴더 내 파일들을 직접 교체하는 방식
"""

import shutil
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pathlib import Path
import os
import datetime
import subprocess

class InternalPatcher:
    def __init__(self):
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 설정"""
        self.root = tk.Tk()
        self.root.title("프로세스4 내부망 패치 도구 v1.0")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="🔧 프로세스4 오류 수정 패치", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 해결 문제 설명
        desc_text = """해결되는 문제:
❌ TopologyException: side location conflict
❌ 좌표계 변환 오류: 'polygons' 
❌ 폴리곤 데이터가 없습니다

✅ 내부망 환경 완벽 지원 (Python 미설치 OK)"""
        
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.LEFT, 
                              background="lightblue", padding="10")
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # 프로세스4 경로 선택
        path_frame = ttk.LabelFrame(main_frame, text="프로세스4 실행파일 위치", padding="10")
        path_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var, width=60)
        path_entry.grid(row=0, column=0, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="찾아보기", command=self.browse_process4)
        browse_btn.grid(row=0, column=1)
        
        # 패치 상태
        status_frame = ttk.LabelFrame(main_frame, text="패치 진행 상태", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_text = tk.Text(status_frame, height=10, width=70)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        status_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        # 진행률 바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        self.check_btn = ttk.Button(button_frame, text="🔍 환경 검사", command=self.check_environment)
        self.check_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.patch_btn = ttk.Button(button_frame, text="🚀 패치 적용", command=self.apply_patch, state=tk.DISABLED)
        self.patch_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.rollback_btn = ttk.Button(button_frame, text="↩️ 롤백", command=self.rollback_patch, state=tk.DISABLED)
        self.rollback_btn.grid(row=0, column=2, padx=(0, 10))
        
        self.exit_btn = ttk.Button(button_frame, text="❌ 종료", command=self.root.quit)
        self.exit_btn.grid(row=0, column=3)
        
        # 초기 메시지
        self.log("🚀 프로세스4 내부망 패치 도구를 시작합니다.")
        self.log("먼저 프로세스4 실행파일을 선택하세요.")
        
    def log(self, message):
        """로그 메시지 출력"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.status_text.insert(tk.END, log_message)
        self.status_text.see(tk.END)
        self.root.update()
        
    def browse_process4(self):
        """프로세스4 실행파일 선택"""
        file_path = filedialog.askopenfilename(
            title="프로세스4 실행파일 선택",
            filetypes=[("실행파일", "*.exe"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            self.path_var.set(file_path)
            self.log(f"📁 선택된 파일: {Path(file_path).name}")
            
    def check_environment(self):
        """환경 검사"""
        self.log("🔍 환경 검사 시작...")
        self.progress_var.set(0)
        
        try:
            # 1. 프로세스4 경로 확인
            process4_path = self.path_var.get()
            if not process4_path:
                raise Exception("프로세스4 실행파일을 선택해주세요")
                
            process4_file = Path(process4_path)
            if not process4_file.exists():
                raise Exception("선택한 프로세스4 파일이 존재하지 않습니다")
                
            self.log(f"✅ 프로세스4 파일 확인: {process4_file.name}")
            self.progress_var.set(20)
            
            # 2. _internal 폴더 확인
            internal_dir = process4_file.parent / "_internal"
            if not internal_dir.exists():
                raise Exception("_internal 폴더를 찾을 수 없습니다. PyInstaller로 빌드된 파일이 아닙니다.")
                
            self.log(f"✅ _internal 폴더 확인: {internal_dir}")
            self.progress_var.set(40)
            
            # 3. 패치 대상 파일 확인
            target_files = {
                "district_road_clipper.py": internal_dir / "src" / "core" / "district_road_clipper.py",
            }
            
            for name, path in target_files.items():
                if path.exists():
                    self.log(f"✅ 대상 파일 발견: {name}")
                else:
                    self.log(f"⚠️ 대상 파일 없음: {name}")
            
            self.progress_var.set(60)
            
            # 4. 패치 소스 파일 확인
            current_dir = Path(__file__).parent
            patch_files = {
                "district_road_clipper.py": "TopologyException 해결",
                "runtime_patch.py": "좌표계 변환 오류 해결"
            }
            
            missing_sources = []
            for filename, description in patch_files.items():
                source_file = current_dir / filename
                if source_file.exists():
                    self.log(f"✅ 패치 파일 확인: {filename} ({description})")
                else:
                    missing_sources.append(filename)
                    self.log(f"❌ 패치 파일 없음: {filename}")
            
            if missing_sources:
                raise Exception(f"패치 파일이 없습니다: {', '.join(missing_sources)}")
            
            self.progress_var.set(80)
            
            # 5. 쓰기 권한 확인
            if not os.access(internal_dir, os.W_OK):
                raise Exception("_internal 폴더에 쓰기 권한이 없습니다. 관리자 권한으로 실행하세요.")
            
            self.progress_var.set(100)
            self.log("🎉 환경 검사 완료! 패치 적용이 가능합니다.")
            self.patch_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"❌ 환경 검사 실패: {str(e)}")
            messagebox.showerror("오류", f"환경 검사 실패:\n{str(e)}")
            
    def apply_patch(self):
        """패치 적용"""
        if not messagebox.askyesno("패치 적용", 
                                  "패치를 적용하시겠습니까?\n\n"
                                  "⚠️ 원본 파일은 자동으로 백업됩니다.\n"
                                  "⚠️ 프로세스4가 실행 중이면 먼저 종료해주세요."):
            return
            
        self.log("🚀 패치 적용 시작...")
        self.progress_var.set(0)
        
        try:
            process4_path = Path(self.path_var.get())
            internal_dir = process4_path.parent / "_internal"
            current_dir = Path(__file__).parent
            
            # 프로세스4 실행 중인지 확인
            if self.is_process4_running():
                if messagebox.askyesno("프로세스 종료", 
                                     "프로세스4가 실행 중입니다.\n자동으로 종료하시겠습니까?"):
                    self.kill_process4()
                else:
                    self.log("❌ 프로세스4를 먼저 종료해주세요")
                    return
            
            # 백업 디렉토리 생성
            backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = process4_path.parent / f"patch_backup_{backup_timestamp}"
            backup_dir.mkdir(exist_ok=True)
            self.log(f"💾 백업 디렉토리 생성: {backup_dir.name}")
            
            # 패치 파일들 정의
            patch_operations = [
                {
                    "source": current_dir / "district_road_clipper.py",
                    "target": internal_dir / "src" / "core" / "district_road_clipper.py",
                    "backup": backup_dir / "district_road_clipper.py",
                    "description": "TopologyException 해결"
                },
                {
                    "source": current_dir / "runtime_patch.py", 
                    "target": internal_dir / "runtime_patch.py",
                    "backup": None,  # 새 파일이므로 백업 불필요
                    "description": "좌표계 변환 오류 해결"
                }
            ]
            
            total_operations = len(patch_operations)
            
            for i, operation in enumerate(patch_operations):
                progress = ((i + 1) / total_operations) * 100
                self.progress_var.set(progress)
                
                source = operation["source"]
                target = operation["target"]
                backup = operation["backup"]
                desc = operation["description"]
                
                self.log(f"📁 처리 중: {source.name} ({desc})")
                
                # 기존 파일 백업
                if target.exists() and backup:
                    shutil.copy2(target, backup)
                    self.log(f"💾 백업 완료: {backup.name}")
                
                # 대상 디렉토리 생성
                target.parent.mkdir(parents=True, exist_ok=True)
                
                # 패치 파일 복사
                shutil.copy2(source, target)
                self.log(f"✅ 패치 적용: {target.name}")
            
            self.progress_var.set(100)
            self.log("🎉 패치 적용 완료!")
            
            # 버튼 상태 업데이트
            self.patch_btn.config(state=tk.DISABLED)
            self.rollback_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("완료", 
                              f"패치가 성공적으로 적용되었습니다!\n\n"
                              f"해결된 문제:\n"
                              f"✅ TopologyException 오류\n"
                              f"✅ 좌표계 변환 오류\n"
                              f"✅ 폴리곤 데이터 누락\n\n"
                              f"백업 위치: {backup_dir.name}\n\n"
                              f"이제 프로세스4를 실행하여 오류가 해결되었는지 확인해보세요.")
            
        except Exception as e:
            self.log(f"❌ 패치 적용 실패: {str(e)}")
            messagebox.showerror("오류", f"패치 적용 실패:\n{str(e)}")
    
    def rollback_patch(self):
        """패치 롤백"""
        if not messagebox.askyesno("롤백", "패치를 롤백하시겠습니까?\n원본 파일로 복구됩니다."):
            return
            
        self.log("↩️ 패치 롤백 시작...")
        
        try:
            process4_path = Path(self.path_var.get())
            internal_dir = process4_path.parent / "_internal"
            
            # 가장 최근 백업 찾기
            backup_dirs = [d for d in process4_path.parent.iterdir() 
                          if d.is_dir() and d.name.startswith("patch_backup_")]
            if not backup_dirs:
                raise Exception("백업 파일을 찾을 수 없습니다")
            
            latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
            self.log(f"📁 백업 디렉토리: {latest_backup.name}")
            
            # 프로세스4 실행 중인지 확인
            if self.is_process4_running():
                if messagebox.askyesno("프로세스 종료", 
                                     "프로세스4가 실행 중입니다.\n자동으로 종료하시겠습니까?"):
                    self.kill_process4()
                else:
                    self.log("❌ 프로세스4를 먼저 종료해주세요")
                    return
            
            # 백업 파일 복원
            rollback_files = [
                {
                    "backup": latest_backup / "district_road_clipper.py",
                    "target": internal_dir / "src" / "core" / "district_road_clipper.py"
                }
            ]
            
            for rollback in rollback_files:
                if rollback["backup"].exists():
                    shutil.copy2(rollback["backup"], rollback["target"])
                    self.log(f"✅ 복원 완료: {rollback['target'].name}")
                else:
                    self.log(f"⚠️ 백업 파일 없음: {rollback['backup'].name}")
            
            # 추가된 파일 제거
            runtime_patch = internal_dir / "runtime_patch.py"
            if runtime_patch.exists():
                runtime_patch.unlink()
                self.log(f"🗑️ 추가 파일 제거: runtime_patch.py")
            
            self.log("🎉 롤백 완료!")
            
            # 버튼 상태 업데이트
            self.patch_btn.config(state=tk.NORMAL)
            self.rollback_btn.config(state=tk.DISABLED)
            
            messagebox.showinfo("완료", "패치가 성공적으로 롤백되었습니다!")
            
        except Exception as e:
            self.log(f"❌ 롤백 실패: {str(e)}")
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
        self.root.mainloop()

if __name__ == "__main__":
    try:
        patcher = InternalPatcher()
        patcher.run()
    except Exception as e:
        messagebox.showerror("오류", f"패치 도구 실행 오류:\n{str(e)}")
