from fpdf import FPDF
import glob

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "1. Agent Routing, RAG Retrieval & Final Response", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Query 1: What are the pricing plans available for your software?", ln=True)
pdf.ln(5)
pdf.image("screenshots/Screenshot 2026-06-27 105547.png", w=190)

# Page 2
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "2. Agent Routing & Final Response Generation", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Query 2: I forgot my account password.", ln=True)
pdf.ln(5)
pdf.image("screenshots/Screenshot 2026-06-27 105608.png", w=190)

# Page 3
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "3. Agent Routing & RAG Retrieval", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Query 3: My application crashes whenever I upload a file.", ln=True)
pdf.ln(5)
pdf.image("screenshots/Screenshot 2026-06-27 105630.png", w=190)

# Page 4
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "4. Human-in-the-Loop Workflow", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Query 4: I need a refund for my annual subscription.", ln=True)
pdf.ln(5)
pdf.image("screenshots/Screenshot 2026-06-27 105707.png", w=190)

# Page 5
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "5. Memory Storage & Recall", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Query 5: What was my previous support issue?", ln=True)
pdf.ln(5)
pdf.image("screenshots/Screenshot 2026-06-27 105733.png", w=190)

pdf.output("screenshots/Screenshots.pdf")
print("PDF created successfully at screenshots/Screenshots.pdf")
