# Face Recognition Using Convolutional Neural Networks

# Tổng quan đề tài
Dự án "Face Recognition Using Convolutional Neural Networks" nhằm phát triển hệ thống nhận diện khuôn mặt dựa trên mô hình CNN. Hệ thống có thể nhận diện khuôn mặt trong hình ảnh và đối chiếu với dữ liệu khuôn mặt lưu trước đó.

# Công nghệ sử dụng

Python: Ngôn ngữ lập trình chính.

TensorFlow/Keras: Xây dựng và huấn luyện mô hình CNN.

OpenCV: Xử lý hình ảnh và video.

dlib: Nhận diện và xác định điểm trên khuôn mặt.

NumPy & Pandas: Xử lý dữ liệu.

# Tập Dữ liệu

Labeled Faces in the Wild (LFW): Đây là tập dữ liệu chính được sử dụng trong nghiên cứu. LFW gồm khoảng 13,000 hình ảnh của 5,749 cá nhân thu thập từ Internet. Tuy nhiên, để đảm bảo mô hình có thể học hiệu quả, chỉ chọn 10 cá nhân có nhiều ảnh nhất, mỗi cá nhân có ít nhất 50 ảnh.

Các ảnh trong LFW được xử lý bằng kỹ thuật deep funneling (căn chỉnh ảnh khuôn mặt) nhằm giảm sự biến đổi trong cùng một lớp (cùng một người) để mạng học được sự khác biệt giữa các lớp tốt hơn.

Vì tập dữ liệu khá lớn khoảng hơn 100 MB nên nhóm chúng tôi sẽ chia sẻ thông qua drive sau : https://drive.google.com/drive/folders/1lw6yyC9m1p26f0TQI0a7VZGQTAq9XW7I?usp=drive_link  

# Mục tiêu nghiên cứu

Mục tiêu chính là sử dụng mạng CNN đã được huấn luyện sẵn (Google’s Inception V3) để thực hiện nhận diện khuôn mặt trên một tập dữ liệu mới (LFW) với số lượng lớp hạn chế (10 người).

Thay vì huấn luyện lại toàn bộ mạng sâu (vốn rất tốn tài nguyên), chúng tôi chỉ huấn luyện lại các lớp trên cùng (top layers) để thích ứng với nhiệm vụ nhận diện khuôn mặt cụ thể.

Mục tiêu là chứng minh khả năng chuyển giao học (transfer learning) của mạng Inception V3 trong việc nhận diện khuôn mặt, đồng thời phân tích hiệu quả của mô hình trên tập dữ liệu thực tế.

# Phương pháp nghiên cứu

Sử dụng mô hình Inception V3 đã được huấn luyện trên ImageNet (1.2 triệu ảnh, 1000 lớp) làm mạng cơ sở.

Thực hiện fine-tuning: chỉ huấn luyện lại lớp cuối cùng hoặc một số lớp trên cùng để thích ứng với dữ liệu mới.

Tiến hành tiền xử lý ảnh bằng cách căn chỉnh khuôn mặt (deep funneling) để giảm nhiễu do góc độ, ánh sáng, phông nền.

Chia dữ liệu thành ba bộ: huấn luyện, validation, kiểm thử để đánh giá mô hình.

Sử dụng các thư viện TensorFlow, matplotlib để xử lý ảnh và huấn luyện mô hình.

Đánh giá hiệu quả mô hình qua các chỉ số như accuracy, precision, recall, F1-score.

# Giả thuyết khoa học

Mạng CNN đã được huấn luyện trên tập dữ liệu lớn (ImageNet) có thể học được các đặc trưng chung của ảnh (hình dạng, màu sắc, kết cấu), vì vậy có thể áp dụng lại cho các bài toán nhận diện khuôn mặt bằng cách huấn luyện lại một phần nhỏ của mạng.

Việc căn chỉnh ảnh khuôn mặt (deep funneling) sẽ giúp giảm biến thể trong cùng một lớp, cải thiện khả năng phân biệt giữa các lớp khác nhau.

Mặc dù số lượng ảnh cho mỗi cá nhân không lớn, mô hình vẫn có thể học để phân biệt các cá nhân khác nhau nếu tập huấn luyện được chuẩn bị hợp lý.

# Đóng góp dự kiến
Cung cấp một ví dụ thực tiễn về cách sử dụng transfer learning với mạng CNN sâu (Inception V3) cho bài toán nhận diện khuôn mặt.

Chia sẻ quy trình tiền xử lý dữ liệu, lựa chọn tập dữ liệu phù hợp và cách xử lý mất cân bằng dữ liệu.

Cung cấp mã nguồn Python minh họa cách tải, xử lý dữ liệu và huấn luyện mô hình, giúp người đọc dễ dàng tái tạo và áp dụng cho các bài toán tương tự.

Đóng góp kiến thức về việc sử dụng mô hình CNN lớn đã huấn luyện sẵn cho các tác vụ nhận diện khuôn mặt cụ thể với tài nguyên hạn chế.



