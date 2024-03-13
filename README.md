# Mô tả
Sử dụng transfer learning trên pretrained model vgg16 sử dụng thư viện pytorch để nhận dạng 2 dữ liệu con ong và con kiến

## Lưu ý
Code đang được cài đặt mặc định vì vậy sẽ train trên cpu. 

## Cài thư viện

    pip install torch torchvision pillow matplotlib 
  
## Chuẩn bị dữ liệu

    python prepare_data.py
  
## Train Model

    python transfer_learning.py
  
## Dự đoán ảnh

Thay đường dẫn của ảnh cần dự đoán trong predict.py:

    python predict.py
