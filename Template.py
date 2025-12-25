css = '''
<style>
.chat-message {
    display: flex;
    margin-bottom: 1rem;
    gap: 0.75rem;
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.bot {
    justify-content: flex-start;
}

.chat-message .avatar {
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 50%;
    overflow: hidden;
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.chat-message .message {
    max-width: 60%;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    word-wrap: break-word;
    line-height: 1.4;
}

.chat-message.user .message {
    background-color: #10a37f;
    color: white;
}

.chat-message.bot .message {
    background-color: #ececf1;
    color: #333;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://api.dicebear.com/7.x/bottts/svg?seed=bot" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://as1.ftcdn.net/jpg/03/64/88/42/1000_F_364884228_JIux2brVPuxvpm7wmgShdUMWkOAQCsXM.jpg" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

