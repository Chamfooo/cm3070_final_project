import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

sender_email = "fp.homesurveillance20@gmail.com"
# TODO: READ FROM ENCRYTPED FILE
password = ""

# Sending the aleat
def send_alert_email(subject, body, image, reciever, sender=sender_email, password=password):
    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = reciever
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    # Attach the image from memory
    if image:
        image.seek(0)  # Rewind the BytesIO object
        img = MIMEImage(image.read(), name='halfway_frame.jpg')
        img.add_header('Content-Disposition', 'attachment', filename='halfway_frame.jpg')
        msg.attach(img)

    # Connect TLS and sending email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        text = msg.as_string()
        server.sendmail(sender, reciever, text)
        server.quit()
        print("Alert email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")