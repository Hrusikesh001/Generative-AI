from pydantic import BaseModel, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str = 'nistish'
    age: Optional[int] = None
    email: EmailStr

new_student = {'age': '20', 'email': 'abc@gmail.com'}

student = Student(**new_student)
print(student)