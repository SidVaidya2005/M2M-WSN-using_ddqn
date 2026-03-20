"""
Legacy web server entry point for backward compatibility.

⚠️  DEPRECATED: Use backend.app instead
    python -m flask --app backend.app run

This file is maintained for backward compatibility only.
"""

from backend.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
