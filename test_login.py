import sys
import os

from app import app, ADMIN_USER, ADMIN_PASS

print("In app.py globals:")
print("ADMIN_USER:", repr(ADMIN_USER))
print("ADMIN_PASS:", repr(ADMIN_PASS))

client = app.test_client()
response = client.post('/admin/login', data={'username': 'admin', 'password': 'admin'})
print("Login status code:", response.status_code)
print("Login Location:", response.headers.get('Location'))
if b'Invalid credentials' in response.data:
    print("Error in HTML: Invalid credentials")
