// $(function() {
//     $("submit_button").on('click', function(e) {
//         e.preventDefault()
//         $.getJSON('/filter/top',
//            {
//         // do nothing
//         });
//     });
// });



document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('button').forEach(button => {
        button.onclick = () => {
            const request = new XMLHttpRequest();
            var form_entry = document.getElementById("entry").value;
            request.open('POST', `/new_entry/${form_entry}`);
            request.onload = () => {
                // response is what the flask function returns
                const response = request.responseText;
                var div = document.createElement("div");
                div.id = "user_chat";
                const node = document.createTextNode(form_entry);
                div.appendChild(node);
                chat_box.appendChild(div);
                // chat_box.scrollBy(0, div.scrollHeight);
            }; 
            request.send();
            chat_box.scrollTop = chat_box.scrollTopMax;
            // chat_box.scrollTo(0, chat_box.scrollHeight - 60);
        };
    });
    
});

var chat_box = document.getElementById("chat_log");
var user_chat = document.getElementById("user_chat");



