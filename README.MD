# ECG Classification
12 Leads ECG classification with LSTM
## Dev
```cuong_tran```
## Install
```buildoutcfg
conda create --name my_env python=3.8
conda activate my_env
pip install -r requirements.txt
```
## Usage
Datasets: https://drive.google.com/file/d/1G7eku6_zdAL2309mTIBn16S-6Q_zTO2F/view?usp=sharing

```buildoutcfg
python train.py --data-path PathToDatasets
python evaluate.py --test-path
```
## Method
- Use basic LSTM module to classify single-lead 
- Comming soon: classification with multi-leads
## Challenges:
(1) Phân lớp dựa trên 1 lead có độ chính xác khá thấp vì bắt bệnh dựa trên cả 12 leads. Hướng tiếp cận: dùng 12 blocks (mỗi block là 1 mô hình LSTM tương tự như single lead), sau đó kết nối 12 block đó lại bằng 1 vài lớp fully connected, đây là hướng tiếp cận đơn giản nhưng khá 'cồng kềnh'  
(2) Output thực tế của bài toán có từ 1-3 class (mắc 1-3 bệnh khác nhau). Hướng tiếp cận 1: nếu người mắc cả bệnh 1 và 2 thì bệnh người này mắc là 12, tức là tách ra bệnh mới, nhưng điều này làm tăng vọt số class output (max là 8x7 + 8x7x6 + 9 = 401 class, thực tế từ phân tích dữ liệu nó rơi vào khoảng 70 class), chia nhỏ lượng data và làm giảm độ chính xác. Hướng tiếp cận 2: mang 2 bệnh thì các leads sẽ mang đặc trưng của  cả 2 bệnh đó, vì vậy chỉ cần chọn ngưỡng xác suất và xác định có bao nhiêu class trên ngưỡng    
(3) Datasets có lượng mẫu trên 1 lead từ 4999-28000 mẫu, với tần số lấy mẫu là 500hz thì thời gian đo điện tim là từ 10-56 giây, nếu input model là 10s thì có thể sẽ dư ra số lẻ, liệu có thể cắt theo kiểu overlap để lấy hết những data này không?
## Evaluate
Single lead ECG:  
Train set:  
<img src="https://user-images.githubusercontent.com/79197775/141642710-8880c40e-7227-496e-82f0-d49648493985.png" alt="..." width="600" />  
Val set:  
<img src="https://user-images.githubusercontent.com/79197775/141642481-39ce0f47-9dab-4959-9519-2f5a4d49d901.png" alt="..." width="600" />  
Test set:  
<img src="https://user-images.githubusercontent.com/79197775/141642771-866f84e5-4e00-4d19-95be-4a196c31ce7a.png" alt="..." width="600" />
