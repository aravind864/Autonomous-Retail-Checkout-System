/**
 * customer_app.js
 * Handles live cart polling and UI updates for the Customer Mobile App.
 */

const CUSTOMER_ID = "CUST_1"; // Replace dynamically in production

async function fetchCart() {
    try {
        const res = await fetch(`/api/cart/${CUSTOMER_ID}`);
        const data = await res.json();

        // --- Update Cart Items ---
        const listEl = document.getElementById('cart-items');
        listEl.innerHTML = '';

        if (data.cart.length === 0) {
            listEl.innerHTML = '<li class="cart-item cart-item--empty">Your cart is empty</li>';
        } else {
            data.cart.forEach(item => {
                listEl.innerHTML += `
                    <li class="cart-item">
                        <span>${item.product_name}</span>
                        <span>₹${item.price}</span>
                    </li>`;
            });
        }

        // --- Update Total ---
        document.getElementById('cart-total').innerText = `₹${data.total}`;

        // --- Update Payment QR Code ---
        const qrSlot = document.getElementById('qr-img-slot');
        if (data.qr_code_base64) {
            qrSlot.innerHTML = `<img src="data:image/png;base64,${data.qr_code_base64}" alt="Payment QR">`;
        } else {
            qrSlot.innerHTML = '<p class="qr-empty">Add items to generate payment QR</p>';
        }

    } catch (e) {
        console.error("Cart sync error:", e);
    }
}

// Auto-refresh cart every 1 second
setInterval(fetchCart, 1000);
fetchCart(); // Initial load
