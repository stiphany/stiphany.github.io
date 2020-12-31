<?php

if(isset($_POST['submit'])) {
 $to = "stiphany@psu.com";
 $subject = $_POST['subject'];
 $name_field = $_POST['name'];
 $email_field = $_POST['email'];
 $message = $_POST['message'];
 
 $body = "From: ".$name_field"\n E-Mail: ".$email_field"\n Message:\n" .$message";
 
if (mail($to, $subject, $body)) {
 echo 'Message sent.';
} else {
  echo 'Rip. Try again.';
}
 
?>
