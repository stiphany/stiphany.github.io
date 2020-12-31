<?php
if(isset($_post['submit'])) {
$to = 'stiphanyt@gmail.com';
$subject = $_post['subject'];
$name_field = $_post['name'];
$email_field = $_post['email'];
$message = $_post['message'];
 
$body = "From: $name_field\n E-Mail: $email_field\n Message:\n $message";
 
echo "Data has been submitted to $to!";
mail($to, $subject, $body);
} else {
echo "blarg!";
}
?>
