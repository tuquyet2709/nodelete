Trích rút sự kiện từ tin tức tiếng Anh

Môi trường : Ubuntu

Ngôn ngữ sử dụng : Python

File thực thi : event_extraction.py, sử dụng tham số dòng lệnh bao gồm train (huấn luyện và tạo model), custom_input (nhập câu từ bàn phím và đưa ra sự kiện cùng các đối số liên quan)

   python event_extraction.py train

   python event_extraction.py custom_input

File kiểm thử độ chính xác của hệ thống test.py

Ngoài các công cụ hỗ trợ đã được import, cần thiết phải tải thêm :
- Bộ dữ liệu ace_2005_td_v7, đưa ra ngoài ngang hàng với thư mục chương trình để có thể chạy các file parser để lấy dữ liệu
- Stanford Parser Full : https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip
(sau khi tải về sửa lại đường dẫn của os.environ['CLASSPATH'] và os.environ['STANFORD_MODELS'] trỏ đến vị trí các file này)
- Stanford Core NLP English : https://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar (sau khi tải về sửa lại đường dẫn của model_path trong hàm dependency_parser trỏ tới model .ser.gz của file này)
