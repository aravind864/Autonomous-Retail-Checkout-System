import qrcode

def calculate_bill(cart_items):
    total = sum(item['price'] for item in cart_items)
    return total

def generate_qr_payment(total_amount, customer_id):
    """Generates a UPI QR code for the calculated bill."""
    if total_amount == 0:
        print("[CHECKOUT] Cart is empty. No payment required.")
        return

    # UPI URL Format: upi://pay?pa=merchant_upi_id&pn=StoreName&am=Amount
    upi_id = "your_store@upi"
    upi_url = f"upi://pay?pa={upi_id}&pn=AutonomousRetail&am={total_amount}"
    
    qr = qrcode.make(upi_url)
    qr.save(f"payment_{customer_id}.png")
    print(f"[CHECKOUT] Bill: ₹{total_amount}. QR Code generated for {customer_id}.")

# Example Execution
# cart = state_manager.get_cart("CUST_1")
# total = calculate_bill(cart)
# generate_qr_payment(total, "CUST_1")