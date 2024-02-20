css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center; /* Add this to vertically center align items */
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    flex-shrink: 0; /* Prevents the avatar from shrinking */
    width: 78px; /* Fixed width */
    height: 78px; /* Fixed height */
    border-radius: 50%;
    overflow: hidden; /* Ensures the image is contained within the border-radius */
}
.chat-message .avatar img {
    width: 100%; /* Make image fill the container */
    height: 100%; /* Make image fill the container */
    object-fit: cover; /* Ensures the image covers the area without distortion */
}
.chat-message .message {
    width: calc(100% - 90px); /* Adjust width to take avatar size and margin into account */
    padding: 0 1.5rem;
    color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <!-- Make sure the path to your image is correct -->
        <img src="data:image/png;base64,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" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

