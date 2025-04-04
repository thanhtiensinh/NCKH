import json
import os
import re
import logging
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường từ file .env
load_dotenv(dotenv_path=".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("API Key không được tìm thấy trong .env")

# Cấu hình Gemini API
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CHAT_HISTORY_FILE = "chat_history.json"

# Hàm tiền xử lý câu hỏi
def preprocess_question(text):
    """Tiền xử lý câu hỏi: chuyển về chữ thường, bỏ dấu câu."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Đọc dữ liệu từ file JSON
def load_json_data(filepath):
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Tệp {filepath} không tồn tại. Tạo tệp mới.")
            save_json_data(filepath, [])  # Tạo file rỗng nếu chưa tồn tại
            return []
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                logger.warning(f"Tệp {filepath} không có định dạng danh sách. Khởi tạo lại.")
                return []
            return data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Lỗi khi đọc tệp {filepath}: {e}")
        return []

# Lưu dữ liệu vào file JSON
def save_json_data(filepath, data):
    try:
        # Kiểm tra quyền truy cập file
        directory = os.path.dirname(filepath) or "."
        if not os.access(directory, os.W_OK):
            raise IOError(f"Không có quyền ghi vào thư mục {directory}")
        
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.debug(f"Đã lưu dữ liệu vào {filepath} thành công.")
    except IOError as e:
        logger.error(f"Lỗi khi lưu tệp {filepath}: {e}")
        raise  # Ném lỗi để dễ phát hiện

# Đọc dữ liệu huấn luyện
def load_trained_data(filepath):
    try:
        if not os.path.exists(filepath):
            logger.error(f"Tệp {filepath} không tồn tại.")
            return {}
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict) and "câu hỏi" in data:
                question_list = data["câu hỏi"]
            elif isinstance(data, list):
                question_list = data
            else:
                logger.error(f"Kỳ vọng một danh sách hoặc từ điển với khóa 'câu hỏi', nhưng nhận được {type(data)}")
                return {}
            if not isinstance(question_list, list):
                logger.error(f"Giá trị của 'câu hỏi' phải là một danh sách, nhưng nhận được {type(question_list)}")
                return {}
            trained_dict = {}
            for item in question_list:
                if not isinstance(item, dict):
                    logger.warning(f"Bỏ qua mục không hợp lệ: {item} (không phải từ điển)")
                    continue
                if "BAN" not in item or "HPU" not in item:
                    logger.warning(f"Bỏ qua mục thiếu khóa: {item}")
                    continue
                trained_dict[item["BAN"]] = item["HPU"]
            logger.info(f"Đã tải dữ liệu huấn luyện với {len(trained_dict)} câu hỏi.")
            return trained_dict
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi đọc tệp trained_data: {e}")
        return {}
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi đọc trained_data: {e}")
        return {}

# Khởi tạo dữ liệu huấn luyện
trained_data = load_trained_data("trained_data.json")

# Chuẩn hóa dữ liệu huấn luyện
processed_trained_data = {preprocess_question(key): value for key, value in trained_data.items()}

# Khởi tạo mô hình Sentence Transformers
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Mã hóa các câu hỏi trong FQA thành vector
question_embeddings = model.encode([preprocess_question(q) for q in list(trained_data.keys())])

# Tìm câu hỏi tương tự bằng Sentence Transformers
def find_best_match_semantic(question, data, threshold=0.6):
    question_embedding = model.encode([preprocess_question(question)])
    cosine_scores = util.cos_sim(question_embedding, question_embeddings)[0]
    best_match_index = cosine_scores.argmax()
    if cosine_scores[best_match_index] > threshold:
        return list(data.keys())[best_match_index]
    return None

# Tìm câu hỏi tương tự bằng rapidfuzz
def find_best_match_fuzzy(question, data, score_cutoff=80):
    processed_question = preprocess_question(question)
    result = process.extractOne(processed_question, list(data.keys()), score_cutoff=score_cutoff)
    if result:
        best_match, _, _ = result
        return best_match
    return None

# Tìm câu hỏi tương tự trong lịch sử hội thoại
def find_best_match_in_history(question, history, score_cutoff=85):
    if not history:
        return None
    # Tạo danh sách các câu hỏi từ lịch sử hội thoại
    history_questions = [entry["question"] for entry in history]
    processed_question = preprocess_question(question)
    result = process.extractOne(processed_question, history_questions, score_cutoff=score_cutoff)
    if result:
        best_match, _, _ = result
        # Tìm câu trả lời tương ứng với câu hỏi khớp nhất
        for entry in history:
            if entry["question"] == best_match:
                return entry["answer"]
    return None

# Lấy câu trả lời
def get_answer(question, data):
    history = load_json_data(CHAT_HISTORY_FILE)
    
    # Bước 1: Tìm kiếm trong trained_data.json bằng rapidfuzz
    best_match_fuzzy = find_best_match_fuzzy(question, processed_trained_data)
    if best_match_fuzzy:
        logger.info(f"Tìm thấy câu hỏi tương tự trong trained_data.json bằng rapidfuzz: {best_match_fuzzy}")
        answer = processed_trained_data.get(best_match_fuzzy)
        save_chat_history(question, answer)
        return answer
    
    # Bước 2: Tìm kiếm trong trained_data.json bằng Sentence Transformers
    best_match_semantic = find_best_match_semantic(question, trained_data)
    if best_match_semantic:
        logger.info(f"Tìm thấy câu hỏi tương tự trong trained_data.json bằng Sentence Transformers: {best_match_semantic}")
        answer = trained_data.get(best_match_semantic)
        save_chat_history(question, answer)
        return answer
    
    # Bước 3: Tìm kiếm trong chat_history.json
    best_match_history = find_best_match_in_history(question, history)
    if best_match_history:
        logger.info(f"Tìm thấy câu hỏi tương tự trong chat_history.json: {question}")
        answer = best_match_history
        save_chat_history(question, answer)
        return answer
    
    # Bước 4: Gọi Gemini API
    logger.info(f"Không tìm thấy câu hỏi tương tự, gọi Gemini API cho câu hỏi: {question}")
    answer = ask_gemini_v2(question, history)
    save_chat_history(question, answer)
    return answer

# Gọi API Gemini với lịch sử hội thoại
def ask_gemini_v2(question, history=None):
    try:
        # Thêm system prompt với thông tin cập nhật về HPU
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction="Bạn là Chatbot HPU, được tạo ra để hỗ trợ sinh viên của trường Đại học Quản lý và Công nghệ Hải Phòng (HPU). HPU hiện có 7 khoa đào tạo các chuyên ngành khác nhau, bao gồm: 1. Khoa Công nghệ thông tin, 2. Khoa Kỹ thuật Điện - Điện tử 3. Khoa Quản trị kinh doanh, 4. Khoa Kỹ thuật môi trường, 5. Khoa Việt Nam học, 6. Khoa Ngôn ngữ Anh, 7. Khoa ngôn ngữ Trung. Tên cũ của trường là Đại học Dân lập Hải Phòng, nhưng hiện tại trường đã chuyển đổi sang loại hình đại học tư thục và đổi tên thành Đại học Quản lý và Công nghệ Hải Phòng từ năm 2019. Hãy trả lời các câu hỏi một cách thân thiện và hữu ích, tập trung vào việc cung cấp thông tin liên quan đến HPU."
        )
        if history is None:
            history = load_json_data(CHAT_HISTORY_FILE)
        
        # Định dạng lịch sử hội thoại theo đúng yêu cầu của Gemini API
        formatted_history = []
        for msg in history:
            if "question" not in msg or "answer" not in msg:
                logger.warning(f"Bỏ qua mục không hợp lệ trong lịch sử hội thoại: {msg}")
                continue
            formatted_history.append({
                "role": "user",
                "parts": [{"text": msg["question"]}]
            })
            formatted_history.append({
                "role": "assistant",
                "parts": [{"text": msg["answer"]}]
            })
        
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(question)
        return response.text.strip() if response else "Không có phản hồi từ Gemini."
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {str(e)}")
        return f"Lỗi khi gọi Gemini API: {str(e)}"

# Lưu lịch sử chat
def save_chat_history(user_question, bot_response):
    try:
        history = load_json_data(CHAT_HISTORY_FILE)
        history.append({"question": user_question, "answer": bot_response})
        save_json_data(CHAT_HISTORY_FILE, history)
        logger.info(f"Đã lưu lịch sử hội thoại: {user_question} -> {bot_response}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu lịch sử hội thoại: {e}")
        raise

# Kết nối MySQL
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="qlsv",
            port=3306
        )
        logger.info(f"Kết nối thành công đến cơ sở dữ liệu: {conn.database}")
        return conn
    except Error as err:
        logger.error(f"Lỗi kết nối cơ sở dữ liệu: {err}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/thongtindangky_new")
def thongtindangky():
    return render_template("thongtindangky_new.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_question = request.json.get("question")
        if not user_question:
            return jsonify({"error": "Không có câu hỏi được cung cấp."}), 400
        
        if not trained_data:
            logger.warning("Không có dữ liệu huấn luyện, thử gọi Gemini API...")
            answer = ask_gemini_v2(user_question)
            save_chat_history(user_question, answer)
            return jsonify({"answer": answer})
            
        answer = get_answer(user_question, trained_data)
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu: {str(e)}")
        return jsonify({"error": f"Lỗi khi xử lý yêu cầu: {str(e)}"}), 500

@app.route("/chat-history", methods=["GET"])
def get_chat_history():
    history = load_json_data(CHAT_HISTORY_FILE)
    return jsonify(history)

@app.route("/register", methods=["POST"])
def register():
    conn = None
    cursor = None
    try:
        user_data = request.json
        logger.info("Dữ liệu nhận được từ frontend: %s", user_data)

        if not isinstance(user_data, dict):
            return jsonify({"error": "Dữ liệu phải là một đối tượng JSON."}), 400

        required_fields = ["name", "phone", "email", "major"]
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        if missing_fields:
            return jsonify({"error": f"Thiếu các trường: {', '.join(missing_fields)}"}), 400

        if len(user_data["name"]) > 100 or len(user_data["email"]) > 100 or len(user_data["major"]) > 100:
            return jsonify({"error": "Tên, email hoặc ngành học vượt quá 100 ký tự."}), 400
        if len(user_data["phone"]) > 20:
            return jsonify({"error": "Số điện thoại vượt quá 20 ký tự."}), 400

        message = user_data.get("message", None)

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến cơ sở dữ liệu."}), 500

        cursor = conn.cursor()
        sql = "INSERT INTO consultation_users (name, phone, email, major, message) VALUES (%s, %s, %s, %s, %s)"
        val = (user_data["name"], user_data["phone"], user_data["email"], user_data["major"], message)
        cursor.execute(sql, val)
        conn.commit()
        
        logger.info(f"Đã chèn {cursor.rowcount} bản ghi.")
        if cursor.rowcount == 0:
            return jsonify({"error": "Không có bản ghi nào được chèn."}), 500

        return jsonify({"message": "Đăng ký thành công!"})
    except Error as err:
        logger.error(f"Lỗi cơ sở dữ liệu: {err}")
        return jsonify({"error": f"Lỗi cơ sở dữ liệu: {str(err)}"}), 500
    except Exception as e:
        logger.error(f"Lỗi khác: {e}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Error as e:
                logger.error(f"Lỗi khi đóng cursor: {e}")
        if conn is not None:
            try:
                conn.close()
            except Error as e:
                logger.error(f"Lỗi khi đóng kết nối: {e}")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)