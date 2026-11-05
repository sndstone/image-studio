import Foundation
import Security

/// Stores the Hugging Face token and the cloud API keys in the user's login keychain.
enum Keychain {
    private static let service = "local.imagestudio"
    static let huggingFaceAccount = "huggingface-token"
    /// One item holds every cloud key (JSON), so an app update asks for Keychain permission once, not per provider.
    static let cloudAccount = "cloud-api-keys"

    private static func baseQuery(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    static func read(account: String = huggingFaceAccount) -> String? {
        var query = baseQuery(account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Like `read`, but a denied or failed read throws instead of looking like "nothing saved".
    static func lookup(account: String) throws -> String? {
        var query = baseQuery(account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        try check(status)
        return (result as? Data).flatMap { String(data: $0, encoding: .utf8) }
    }

    static func save(_ value: String, account: String = huggingFaceAccount) throws {
        let attributes = [kSecValueData as String: Data(value.utf8)]
        var status = SecItemUpdate(baseQuery(account) as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var query = baseQuery(account)
            query[kSecValueData as String] = Data(value.utf8)
            status = SecItemAdd(query as CFDictionary, nil)
        }
        try check(status)
    }

    static func delete(account: String = huggingFaceAccount) throws {
        let status = SecItemDelete(baseQuery(account) as CFDictionary)
        if status != errSecItemNotFound { try check(status) }
    }

    private static func check(_ status: OSStatus) throws {
        guard status != errSecSuccess else { return }
        let message = SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)"
        throw BackendError(message: message)
    }
}
