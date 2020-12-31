<?php

if(isset($_POST['submit'])) {
 $name = $_POST['name'];
 $subject = $_POST['subject'];
 $mailFrom = $_POST['email'];
 $message = $_POST['message'];
 
 $to = "stiphany@psu.com";
 $headers = "From ".$mailFrom;
 $body = "Email from ",$name.".\n\n".$message;

 mail($to, $subject, $body, $headers);
 header("Location: index.php?mailsend");
}
 
?>
