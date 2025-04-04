import pandas as pd
import json
import os

# Định nghĩa đường dẫn file
EXCEL_FILE = "training1.xlsx"  # 🔹 Thay bằng tên file Excel của bạn
JSON_FILE = "data.json"  # 🔹 File JSON đầu ra

def đọc_dữ_liệu_excel():
    """ Đọc file Excel và chuyển đổi thành danh sách câu hỏi - câu trả lời """
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Lỗi: Không tìm thấy file {EXCEL_FILE}")
        return None

    try:
        df = pd.read_excel(EXCEL_FILE)

        # Kiểm tra xem các cột cần thiết có tồn tại không
        if "Bạn" not in df.columns or "HPU" not in df.columns:
            print("❌ Lỗi: File Excel phải có cột 'Bạn' và 'HPU'!")
            return None

        # Kiểm tra xem file có dữ liệu không
        if df.empty:
            print("⚠️ Cảnh báo: File Excel không có dữ liệu!")
            return None

        # Chuyển đổi dữ liệu thành danh sách từ điển
        data = {"câu hỏi": df.to_dict(orient="records")}
        return data

    except Exception as e:
        print(f"❌ Lỗi khi đọc file Excel: {e}")
        return None

def lưu_dữ_liệu_vào_json():
    """ Đọc dữ liệu từ Excel và lưu vào JSON """
    data = đọc_dữ_liệu_excel()

    if data:
        try:
            with open(JSON_FILE, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"✅ Đã lưu dữ liệu vào {JSON_FILE} thành công!")

            # 🔍 Hiển thị 5 câu hỏi đầu tiên để kiểm tra
            print("\n🔹 Một số dữ liệu từ file Excel:")
            for item in data["câu hỏi"][:5]:  # Hiển thị 5 câu hỏi đầu tiên
                print(f"- Bạn: {item['Bạn']}\n  HPU: {item['HPU']}\n")

        except Exception as e:
            print(f"❌ Lỗi khi lưu JSON: {e}")

# Gọi hàm để thực hiện chuyển đổi khi chạy script
if __name__ == "__main__":
    lưu_dữ_liệu_vào_json()