from asyncio import Queue
from typing import Awaitable, Any

from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform import Platform, AstrMessageEvent, PlatformMetadata, AstrBotMessage, MessageType, \
    MessageMember
from astrbot.core.platform.message_session import MessageSesion
from astrbot.core.platform.sources.wechat_websocket.wechat_websocket_message_event import WeChatWebsocketMessageEvent

# TODO: 完善代码
class WeChatWebsocketAdapter(Platform):

    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: Queue
    ) -> None:
        super().__init__(event_queue)
        self.config = platform_config

        self.unique_session = platform_settings["unique_session"]

        self.client_id = platform_config["client_id"]
        self.client_secret = platform_config["client_secret"]

    def run(self) -> Awaitable[Any]:
        pass

    async def terminate(self):
        return await super().terminate()

    def meta(self) -> PlatformMetadata:
        return PlatformMetadata(
            name="wechat-websocket",
            description="微信机器人个人适配",
            id=self.config.get("id")
        )

    async def send_by_session(self,
                              session: MessageSesion,
                              message_chain: MessageChain):
        dummy_message_obj = AstrBotMessage()
        dummy_message_obj.session_id = session.session_id
        # 根据 session_id 判断消息类型
        if "@chatroom" in session.session_id:
            dummy_message_obj.type = MessageType.GROUP_MESSAGE
            if "#" in session.session_id:
                dummy_message_obj.group_id = session.session_id.split("#")[0]
            else:
                dummy_message_obj.group_id = session.session_id
            dummy_message_obj.sender = MessageMember(user_id="", nickname="")
        else:
            dummy_message_obj.type = MessageType.FRIEND_MESSAGE
            dummy_message_obj.group_id = ""
            dummy_message_obj.sender = MessageMember(user_id="", nickname="")
        sending_event = WeChatWebsocketMessageEvent(
            message_str="",
            message_obj=dummy_message_obj,
            platform_meta=self.meta(),
            session_id=session.session_id,
            adapter=self,
        )
        await sending_event.send(message_chain)

    def commit_event(self, event: AstrMessageEvent):
        super().commit_event(event)

    def get_client(self):
        super().get_client()
