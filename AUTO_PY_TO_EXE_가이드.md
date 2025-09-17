# 🎯 auto-py-to-exe GUI 사용 가이드

## 🚀 **GUI가 열렸나요?**
**웹 브라우저에서 `http://localhost:4000` 주소가 열려야 합니다.**

---

## 📋 **단계별 설정 방법**

### **1️⃣ 기본 설정**

**Script Location:**
```
📁 I:\gcp_rl\process3_inference.py
```
→ **"Browse"** 버튼 클릭해서 `process3_inference.py` 선택

**Onefile:**
```
☑️ One File   (추천 - 배포 쉬움)
☐ One Directory
```

**Console Window:**
```
☐ Console Based   
☑️ Window Based (No Console)   (GUI 앱이므로)
```

---

### **2️⃣ 아이콘 설정 (선택사항)**

**Icon:**
```
아이콘 파일(.ico)이 있다면 추가
없으면 비워둠
```

---

### **3️⃣ 고급 설정 (중요!)**

**Additional Files 탭:**
```
📁 추가할 폴더들:
✅ src → src
✅ configs → configs  
✅ src/learning/models → src/learning/models
```

각각 **"Add Folder"** 클릭해서 추가:
1. `src` 폴더 → **destination**: `src`
2. `configs` 폴더 → **destination**: `configs`
3. `src/learning/models` → **destination**: `src/learning/models`

**Hidden Imports 탭:**
```
추가할 모듈들 (줄바꿈으로 구분):

PyQt5.QtCore
PyQt5.QtGui
PyQt5.QtWidgets
PyQt5.sip
geopandas
fiona
fiona.drvsupport
shapely
shapely.geometry
pyproj
torch
torch.nn
numpy
pandas
scipy
sklearn
networkx
cv2
src.core
src.learning
src.process3
src.ui
src.utils
src.filters
```

**Excluded Modules 탭:**
```
제외할 모듈들 (파일 크기 줄이기):

tkinter
matplotlib.tests
numpy.tests
pandas.tests
PIL.tests
```

---

### **4️⃣ 고급 옵션**

**Advanced 탭:**
```
UPX: ☑️ Use UPX (파일 압축)
Optimize: 0 (기본값)
```

---

### **5️⃣ 실행하기**

**하단의 큰 파란 버튼 클릭:**
```
🚀 CONVERT .PY TO .EXE
```

---

## 📊 **빌드 과정 모니터링**

### **로그 화면에서 확인:**
```
✅ Building... 
✅ Processing...
✅ Analyzing dependencies...
✅ Creating executable...
✅ Build completed successfully!
```

### **오류 발생 시:**
```
❌ Missing module 오류
→ Hidden Imports에 해당 모듈 추가

❌ File not found 오류  
→ Additional Files에 파일/폴더 추가

❌ Memory 오류
→ One Directory 방식으로 변경
```

---

## 📁 **결과 확인**

### **성공 시 생성되는 파일:**
```
📂 output/
└── 🚀 process3_inference.exe   (약 500MB~2GB)
```

### **실행 테스트:**
```bash
# 더블클릭하거나 명령줄에서:
cd output
process3_inference.exe
```

---

## 🔧 **문제 해결**

### **pathlib 오류:**
```bash
# 이미 제거 중이지만, 수동으로도 가능:
pip uninstall pathlib -y
```

### **PyQt5 오류:**
```bash
pip uninstall PyQt5
pip install PyQt5==5.15.9
```

### **메모리 부족:**
```
GUI에서:
☐ One File 
☑️ One Directory   (이것으로 변경)
```

### **모듈 누락:**
```
Hidden Imports 탭에서 누락된 모듈 추가:
예: ModuleNotFoundError: requests
→ Hidden Imports에 "requests" 추가
```

---

## 🎯 **빠른 설정 템플릿**

### **최소 설정 (빠른 테스트용):**
1. Script: `process3_inference.py`
2. One File: ☑️
3. Window Based: ☑️
4. Hidden Imports: `PyQt5.QtCore, PyQt5.QtGui, PyQt5.QtWidgets`
5. **CONVERT** 클릭!

### **완전 설정 (배포용):**
1. 위의 전체 가이드 따라하기
2. 모든 폴더와 모듈 추가
3. UPX 압축 활성화
4. **CONVERT** 클릭!

---

## 🚀 **auto-py-to-exe 장점**

✅ **GUI 인터페이스** - 클릭만으로 설정  
✅ **실시간 미리보기** - 설정이 바로 반영됨  
✅ **에러 표시** - 문제점을 바로 알려줌  
✅ **설정 저장** - JSON으로 설정 저장 가능  
✅ **로그 확인** - 빌드 과정 실시간 확인  

---

## 💡 **추가 팁**

### **설정 저장하기:**
```
Settings → Save Configuration
→ 나중에 Load Configuration으로 재사용
```

### **빌드 시간 단축:**
```
1. One Directory 방식 선택
2. 불필요한 모듈 제외
3. 작은 테스트부터 시작
```

### **배포 패키지 만들기:**
```
📦 최종_배포/
├── 🚀 process3_inference.exe
├── 📖 사용법.txt
├── 📂 필수폴더/
└── 📜 install.bat
```

---

**🎉 GUI가 열렸다면 위의 가이드대로 설정하세요!**  
**더 쉽고 직관적으로 EXE 파일을 만들 수 있습니다!** 🚀 