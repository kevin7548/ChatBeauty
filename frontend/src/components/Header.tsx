import "./Header.css";

interface HeaderProps {
  onLogoClick: () => void;
}

export function Header({ onLogoClick }: HeaderProps) {
  return (
    <header className="header">
      <div className="logo" onClick={onLogoClick} style={{ cursor: "pointer" }}>
        <div className="logo-text">
          <span className="chat">Chat</span>
          <span className="beauty">Beauty</span>
        </div>
        <span className="sparkle">{"✨"}</span>
      </div>
      <span className="header-badge">AI Powered</span>
    </header>
  );
}
