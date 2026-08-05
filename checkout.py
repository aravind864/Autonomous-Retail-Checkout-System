import io
import base64
import qrcode

def calculate_bill(cart_items):
    total = sum(item.get('price', 0) for item in cart_items)
    return total

def generate_qr_payment(total_amount, customer_id):
    """Generates a UPI QR code for the calculated bill and saves to disk."""
    if total_amount == 0:
        print("[CHECKOUT] Cart is empty. No payment required.")
        return None

    upi_id = "your_store@upi"
    upi_url = f"upi://pay?pa={upi_id}&pn=AutonomousRetail&am={total_amount}"
    
    qr = qrcode.make(upi_url)
    filename = f"payment_{customer_id}.png"
    qr.save(filename)
    print(f"[CHECKOUT] Bill: ₹{total_amount}. QR Code saved to {filename}.")
    return filename

def generate_qr_base64(total_amount, customer_id):
    """Generates a UPI QR code as a base64 string for direct web display."""
    if total_amount == 0:
        return ""

    upi_id = "your_store@upi"
    upi_url = f"upi://pay?pa={upi_id}&pn=AutonomousRetail&am={total_amount}"
    
    qr = qrcode.make(upi_url)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"