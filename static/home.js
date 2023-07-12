var chat_box = document.getElementById("chat_log");
var user_chat = document.getElementById("user_chat");

function Scroller (){
    chat_box.scrollBy(0, user_chat.scrollHeight);
}
