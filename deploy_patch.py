#!/usr/bin/env python3
"""
프로세스4 패치 배포 도구
런타임 패치 파일을 프로세스4 실행파일 위치에 복사
"""

import shutil
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pathlib import Path
import os

class PatchDeployer:
    def __init__(self):
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 설정"""
        self.root = tk.Tk()
        self.root.title("프로세스4 패치 배포 도구")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="🔧 프로세스4 오류 수정 패치 배포", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 설명
        desc_text = """이 도구는 프로세스4의 다음 오류들을 해결합니다:

❌ TopologyException: side location conflict
❌ 좌표계 변환 오류: 'polygons' 
❌ 폴리곤 데이터가 없습니다

패치 방식: 런타임 패치 (실행 시 자동 적용)"""
        
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.LEFT)
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=tk.W)
        
        # 프로세스4 경로 선택
        path_frame = ttk.LabelFrame(main_frame, text="프로세스4 실행파일 위치", padding="10")
        path_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var, width=50)
        path_entry.grid(row=0, column=0, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="찾아보기", command=self.browse_process4)
        browse_btn.grid(row=0, column=1)
        
        # 패치 파일 상태
        status_frame = ttk.LabelFrame(main_frame, text="패치 파일 상태", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_text = tk.Text(status_frame, height=8, width=60)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        status_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
        
        self.check_btn = ttk.Button(button_frame, text="🔍 환경 검사", command=self.check_environment)
        self.check_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.deploy_btn = ttk.Button(button_frame, text="🚀 패치 배포", command=self.deploy_patch, state=tk.DISABLED)
        self.deploy_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.test_btn = ttk.Button(button_frame, text="🧪 테스트", command=self.test_patch, state=tk.DISABLED)
        self.test_btn.grid(row=0, column=2, padx=(0, 10))
        
        self.exit_btn = ttk.Button(button_frame, text="❌ 종료", command=self.root.quit)
        self.exit_btn.grid(row=0, column=3)
        
        # 초기 메시지
        self.log("프로세스4 패치 배포 도구를 시작합니다.")
        self.log("먼저 프로세스4 실행파일 위치를 선택하세요.")
        
    def log(self, message):
        """로그 메시지 출력"""
        self.status_text.insert(tk.END, f"{message}\n")
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
            self.log(f"선택된 파일: {file_path}")
            
    def check_environment(self):
        """환경 검사"""
        self.log("🔍 환경 검사 시작...")
        
        try:
            # 1. 프로세스4 경로 확인
            process4_path = self.path_var.get()
            if not process4_path:
                raise Exception("프로세스4 실행파일을 선택해주세요")
                
            process4_file = Path(process4_path)
            if not process4_file.exists():
                raise Exception("선택한 프로세스4 파일이 존재하지 않습니다")
                
            self.log(f"✅ 프로세스4 파일 확인: {process4_file.name}")
            
            # 2. 패치 파일 확인
            current_dir = Path(__file__).parent
            patch_files = {
                "runtime_patch.py": "런타임 패치 모듈",
                "process4_inference.py": "수정된 메인 모듈"
            }
            
            missing_files = []
            for filename, description in patch_files.items():
                file_path = current_dir / filename
                if file_path.exists():
                    self.log(f"✅ {description}: {filename}")
                else:
                    missing_files.append(filename)
                    self.log(f"❌ 누락: {filename}")
            
            if missing_files:
                raise Exception(f"패치 파일이 누락되었습니다: {', '.join(missing_files)}")
            
            # 3. 대상 디렉토리 확인
            target_dir = process4_file.parent
            self.log(f"✅ 배포 대상 디렉토리: {target_dir}")
            
            if not os.access(target_dir, os.W_OK):
                raise Exception("대상 디렉토리에 쓰기 권한이 없습니다. 관리자 권한으로 실행하세요.")
            
            self.log("🎉 환경 검사 완료! 패치 배포가 가능합니다.")
            self.deploy_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"❌ 환경 검사 실패: {str(e)}")
            messagebox.showerror("오류", f"환경 검사 실패:\n{str(e)}")
            
    def deploy_patch(self):
        """패치 배포"""
        if not messagebox.askyesno("패치 배포", 
                                  "패치를 배포하시겠습니까?\n\n"
                                  "⚠️ 기존 파일이 있으면 백업 후 교체됩니다."):
            return
            
        self.log("🚀 패치 배포 시작...")
        
        try:
            process4_path = Path(self.path_var.get())
            target_dir = process4_path.parent
            current_dir = Path(__file__).parent
            
            # 백업 디렉토리 생성
            backup_dir = target_dir / "patch_backup"
            backup_dir.mkdir(exist_ok=True)
            
            # 패치 파일 복사
            patch_files = ["runtime_patch.py", "process4_inference.py"]
            
            for filename in patch_files:
                source_file = current_dir / filename
                target_file = target_dir / filename
                
                # 기존 파일 백업
                if target_file.exists():
                    backup_file = backup_dir / f"{filename}.backup"
                    shutil.copy2(target_file, backup_file)
                    self.log(f"💾 백업: {filename} → {backup_file.name}")
                
                # 패치 파일 복사
                shutil.copy2(source_file, target_file)
                self.log(f"✅ 배포: {filename}")
            
            self.log("🎉 패치 배포 완료!")
            self.log(f"백업 위치: {backup_dir}")
            
            self.test_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("완료", 
                              f"패치가 성공적으로 배포되었습니다!\n\n"
                              f"이제 프로세스4를 실행하여 오류가 해결되었는지 확인하세요.\n\n"
                              f"백업 위치: {backup_dir}")
            
        except Exception as e:
            self.log(f"❌ 패치 배포 실패: {str(e)}")
            messagebox.showerror("오류", f"패치 배포 실패:\n{str(e)}")
            
    def test_patch(self):
        """패치 테스트"""
        process4_path = self.path_var.get()
        if not process4_path:
            messagebox.showwarning("경고", "프로세스4 경로가 설정되지 않았습니다")
            return
            
        if messagebox.askyesno("테스트", "프로세스4를 실행하여 패치를 테스트하시겠습니까?"):
            try:
                import subprocess
                subprocess.Popen([process4_path])
                self.log("🧪 프로세스4 테스트 실행 중...")
                messagebox.showinfo("테스트", "프로세스4가 실행되었습니다.\n오류가 해결되었는지 확인해보세요.")
            except Exception as e:
                self.log(f"❌ 테스트 실행 실패: {str(e)}")
                messagebox.showerror("오류", f"테스트 실행 실패:\n{str(e)}")
    
    def run(self):
        """배포 도구 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        deployer = PatchDeployer()
        deployer.run()
    except Exception as e:
        messagebox.showerror("오류", f"배포 도구 실행 오류:\n{str(e)}")
