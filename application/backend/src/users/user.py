from passlib.apps import custom_app_context as pwd_context


class User:

    def __init__(self, username, hashed_pwd=None):
        self.username = username
        self.password_hash = hashed_pwd

    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)