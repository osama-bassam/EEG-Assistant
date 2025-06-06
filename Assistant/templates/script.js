// script.js
function sendMessage() {
  const name = document.getElementById("name").value;
  const message = document.getElementById("message").value;

  if (name && message) {
    alert(`Thank you, ${name}! Your message has been sent.`);
    document.getElementById("name").value = '';
    document.getElementById("message").value = '';
  } else {
    alert("Please fill in all fields.");
  }
}
